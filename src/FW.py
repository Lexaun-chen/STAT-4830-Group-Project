import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init



def get_offer_data(data_para, num_limit):
    offerset_list = []
    sell_list = []
    mask_list = []
    max_num = 39
    n = 0
    for srch_id, group in data_para:
        if n == num_limit:
            break
        num_product = len(group)
        if len(group) >= max_num:
            print('invalid maximum')
        if (group['price_usd'] > 1000).any():
            continue
        if (group['srch_booking_window'] > 365).any():
            continue
        # if (group['random_bool'] == 0).any():
        #     continue
        offerset = group.drop(columns=['booking_bool', 'srch_id', 'srch_booking_window']).values
        offer_dummy = np.zeros((max_num - num_product, offerset.shape[1]))
        offerset = np.vstack((offerset, offer_dummy))

        offer_mask = np.append(np.ones(num_product + 1), np.zeros(max_num - num_product - 1))

        if group['booking_bool'].sum() == 0:
            num_sell = np.append(group['booking_bool'].values, 1).reshape(1, -1)
        else:
            num_sell = np.append(group['booking_bool'].values, 0).reshape(1, -1)

        offerset_list.append(offerset)
        sell_list.append(num_sell)
        mask_list.append(offer_mask)
        n += 1
    offerset_list = np.array(offerset_list)
    mask_list = np.array(mask_list)
    return offerset_list, sell_list, mask_list



if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available:", device)
    torch.cuda.init()
else:
    device = torch.device("cpu")
    print("GPU unavailable: CPU")


class Sales:
    def __init__(self, all_offer_sets, sell_num, mask):
        # all_offer_sets: list [offerset1: (num_products, num_features), offerset2: (num_products, num_features)...]
        # sell_num: nd.array (num_offers, num_products)
        # get the number of offers in a sales dataset
        self.offer_set_list = []
        self.fw = None
        self.N_sales = torch.tensor(np.concatenate(sell_num, axis=1), dtype=torch.float64, device=device)
        self.offer_feature = torch.tensor(all_offer_sets, dtype=torch.float64)
        self.original_shape = self.offer_feature.shape
        self.feature_concat = self.offer_feature.reshape(-1, self.offer_feature.shape[2]).to(device)
        self.N = torch.sum(self.N_sales)
        self.mask = torch.tensor(mask, dtype=torch.float64, device=device)
        self.mask_flat = self.mask.reshape((-1,))
        self.masked_feature_concat = self.feature_concat[self.mask_flat == 1]

    def calculate_all_choice_prob(self, W):
        # W: rule-weight taste vector (|Ruleset|, 1)
        # ruleset: Rule object
        rule_feature = self.feature_concat
        self.fw = self.calculate_choice_prob(rule_feature, W)
        return self.fw

    def calculate_choice_prob(self, rule_feature, W):
        """output choice probability tensor for every product"""
        # W: rule-weight taste vector (|Ruleset|, 1)
        # Z:  (｜St｜, |Ruleset|) * (|Ruleset|, 1) -> (｜St｜, 1)
        Z = torch.matmul(rule_feature, W).reshape(self.original_shape[:2])
        masked_Z = Z.masked_fill(self.mask == 0, float('-inf'))
        softmax_result = F.softmax(masked_Z, dim=-1).reshape((-1, 1))[self.mask_flat == 1]
        return softmax_result


class Problem_FW:
    def __init__(self, S, N, M):
        # S: [St] the list of offerset N: [Njt] the sales of jth product in St offer set
        self.alpha = torch.ones(1, dtype=torch.float64, device=device)
        self.soft_RMSE = None
        self.hard_RMSE = None
        self.fw_estimate = None
        self.sales = Sales(S, N, M)
        # Define and initialize the ruleset
        self.feature_num = S.shape[-1]
        # Put the feature to the GPU
        self.sales.feature_concat = self.sales.feature_concat.to(device)
        # Define a consumer list that contain consumer types
        self.consumer_list = []
        # Define the main problem NLL loss
        self.NLL_main = None
        # Define the current likelihood convex combination
        self.g = None
        # Define the current likelihood gradient for support finding
        self.NLL_gradient = None
        # Define a list to contain all fw choice likelihood
        self.fw_list = []
        # Define a new sales data for further estimation
        self.sales_estimate = None

    def initialize(self):
        # 1. We initialize a taste vector W
        W = torch.empty((self.feature_num, 1), dtype=torch.float64, device=device, requires_grad=True)
        a = -1e-5
        b = 1e-5
        init.uniform_(W, a, b)
        # 2. Add the initial consumer type in the consumer list
        initial_type = ConsumerType(
            W,
            torch.tensor([1], dtype=torch.float64, requires_grad=True, device=device),
            self.sales)
        self.consumer_list.append(initial_type)
        self.fw_list.append(initial_type.fw)
        # 4. Update the main problem NLL loss
        self.main_problem_loss()

    def main_problem_loss(self):
        """Here we calculate the NLL Loss for the main problem"""
        N = self.sales.N
        normalize_term = torch.tensor(1 / N, dtype=torch.float64, device=device)
        N_sales = self.sales.N_sales
        # Now we get the combination of the consumer choice likelihood
        # this step can also be done by inner product by setting a tensor in problem to store all consumer data
        f = torch.zeros(N_sales.shape, dtype=torch.float64, device=device).t()
        for ct in self.consumer_list:
            f += ct.alpha * ct.fw
        f_log = torch.log(f)
        self.NLL_main = -normalize_term * torch.matmul(N_sales, f_log)
        # calculate the current g
        self.g = f
        # Update the gradient for the next support finding step
        with torch.no_grad():
            self.NLL_gradient = -normalize_term * torch.mul(1 / self.g.t(), N_sales)
        return self.NLL_main

    def support_finding_loss(self, W):
        """Here we calculate the Loss for the support finding step"""
        fw = self.sales.calculate_all_choice_prob(W)
        fw_log = torch.log(fw)
        return torch.matmul(self.NLL_gradient, fw_log)

    def proportion_update_loss(self, alpha):
        N = self.sales.N
        alpha = F.softmax(alpha, dim=0)
        normalize_term = torch.tensor(1 / N, dtype=torch.float64, device=device)
        N_sales = self.sales.N_sales
        with torch.no_grad():
            fw_tensor = torch.cat(self.fw_list, dim=1)
        f = torch.matmul(fw_tensor, alpha)
        f_log = torch.log(f)
        return -normalize_term * torch.matmul(N_sales, f_log)

    def search_for_next_consumer_type(self):
        print('-----Consumer Type Search Begin-----')
        # Initialize the taste vector w
        W = torch.empty((self.feature_num, 1), dtype=torch.float64, device=device, requires_grad=True)
        a = -1e-1
        b = 1e-1
        init.uniform_(W, a, b)
        # The loss in the support find step i.e. the sub-problem in conditional gradient descent
        print('-----Rule Weight Optimization-----')
        loss_previous = 1e100
        optimizer = optim.Adam([W], lr=5e-5)
        optimizer.zero_grad()
        w_previous = None
        for epoch in range(20000):
            loss = self.support_finding_loss(W)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{2000}], Loss: {loss.item():.4f}')
            loss_current = loss.item()
            if loss_current < loss_previous:
                loss_previous = loss_current
                with torch.no_grad():
                    w_previous = W
            else:
                W = w_previous.requires_grad_(True)
                break

        new_consumer = ConsumerType(
            W,
            torch.tensor([0], dtype=torch.float64, device=device),
            self.sales)
        self.consumer_list.append(new_consumer)
        self.fw_list.append(new_consumer.fw)
        print('-----Consumer Type Search End-----')

    def proportion_update(self):
        print('-----Proportion Update Search Begin-----')
        # new_alpha = torch.zeros(1, requires_grad=True, dtype=torch.float64, device=device)
        # with torch.no_grad():
        #     alpha = torch.concatenate((self.alpha, new_alpha))
        # alpha = alpha.requires_grad_(True)
        alpha = torch.empty((len(self.consumer_list), 1), dtype=torch.float64, device=device, requires_grad=True)
        a = -1e-5
        b = 1e-5
        init.uniform_(alpha, a, b)
        optimizer = optim.Adam([alpha], lr=5e-2)
        optimizer.zero_grad()
        loss_previous = 1e100
        alpha_previous = None
        for epoch in range(50000):
            loss = self.proportion_update_loss(alpha)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{50000}], Loss: {loss.item():.4f}')
            loss_current = loss.item()
            if loss_current < loss_previous:
                loss_previous = loss_current
                with torch.no_grad():
                    alpha_previous = alpha
            else:
                alpha = alpha_previous.requires_grad_(True)
                break
        # self.alpha = alpha
        alpha = F.softmax(alpha, dim=0)
        for i, consumer in enumerate(self.consumer_list):
            consumer.alpha = alpha[i]
        self.main_problem_loss()
        print('-----Proportion Update Search End-----')

    def estimate_NLL(self, sales_for_estimate, N_estimate, mask_for_estimate):
        self.sales_estimate = Sales(sales_for_estimate, N_estimate, mask_for_estimate)
        N = self.sales_estimate.N
        normalize_term = torch.tensor(1 / N, dtype=torch.float64, device=device)
        N_sales = self.sales_estimate.N_sales
        f = torch.zeros(N_sales.shape, dtype=torch.float64, device=device).t()
        for ct in self.consumer_list:
            f += ct.alpha * self.sales_estimate.calculate_all_choice_prob(ct.taste)
        f_log = torch.log(f)
        return -normalize_term * torch.matmul(N_sales, f_log), f

    def estimate_rank(self, sales_for_estimate, N_estimate, mask_for_estimate):
        rank_list = []
        for o, s, msk in zip(sales_for_estimate, N_estimate, mask_for_estimate):
            prediction = self.estimate_NLL(np.array([o]),
                                           np.array([s]),
                                           np.array([msk]))[1].flatten().detach().cpu().numpy()
            choice = np.argmax(s)
            value_at_index = prediction[choice]
            sorted_prediction = np.sort(prediction)
            rank = np.searchsorted(sorted_prediction, value_at_index, side='right')
            percentile_rank = (1 - (rank / len(s[0]))) * 100
            rank_list.append(percentile_rank)
        return rank_list


class ConsumerType:
    def __init__(self, weight, alpha, problem_sales):
        # rule_weight (tensor): the weight of each rule in the consideration rules (|consideration_rule|, 1)
        self.taste = weight
        # alpha (float.64): the proportion of this type in the distribution
        self.alpha = alpha
        # fw: the choice likelihood vector for the consumer type
        # problem_sales: the sales object for this problem
        self.fw = problem_sales.calculate_all_choice_prob(self.taste)


def train_frank_wolfe(type_num, tr_offerset_list, tr_sell_list, tr_mask_list):
    problem = Problem_FW(tr_offerset_list, tr_sell_list, tr_mask_list)
    problem.initialize()
    n = type_num
    NLL_LOSS_LIST = [problem.NLL_main.item()]
    for m in range(n):
        print('-----Consumer Type ' + str(m + 1) + ' Start Searching------')
        problem.search_for_next_consumer_type()
        problem.proportion_update()
        print('main problem loss: ', problem.NLL_main.item())
        print('-----One Iteration Done-----')
        NLL_LOSS_LIST.append(problem.NLL_main.item())
    for i in range(1, n + 1):
        print('-----Consumer Type ' + str(i) + '------')
        print('Consumer Proportion:', problem.consumer_list[i].alpha.item())
    return problem, NLL_LOSS_LIST


# MNL_problem, NLL_LIST = train_frank_wolfe(10)
# print(MNL_problem.estimate(te_offerset_list, te_sell_list, te_mask_list))
# plt.figure(figsize=(10, 6))
# plt.plot(NLL_LIST, marker='o', linestyle='-', color='b', label='FW')
# plt.title('NLL LOSS VALUE')
# plt.xlabel('ITERATION')
# plt.ylabel('NLL LOSS')
# plt.legend()
# plt.grid(True)
# plt.show()
# fw, _ = train_frank_wolfe(20)
# print('tr_loss', fw.NLL_main, 'te_loss', fw.estimate_NLL(te_offerset_list, te_sell_list, te_mask_list)[0])
# tr_ranking = np.array(fw.estimate_rank(tr_offerset_list, tr_sell_list, tr_mask_list))
# te_ranking = np.array(fw.estimate_rank(te_offerset_list, te_sell_list, te_mask_list))
# acc_tr = np.sum((tr_ranking <= 20).astype(int)) / len(tr_ranking)
# acc_te = np.sum((te_ranking <= 20).astype(int)) / len(te_ranking)
# print('Train ACC:', acc_tr, 'Test ACC:', acc_te)
