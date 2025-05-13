import numpy as np
import itertools
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch


"""
1. Get Synthetic Hypothetical Data
"""

def generate_one_hot(probabilities):
    probabilities = np.array(probabilities)
    p_index = np.random.choice(len(probabilities), p=probabilities)
    one_hot = np.zeros_like(probabilities)
    one_hot[p_index] = 1
    return one_hot


product_set = [0, 1, 2, 3]
offer_set = [0, 1, 2, 3]
X = []
Y = []


hypothetical_choice_p = [[0.98, 0.02, 0, 0],
                         [0.5, 0, 0.5, 0],
                         [0.5, 0, 0, 0.5],
                         [0, 0.5, 0.5, 0],
                         [0, 0.5, 0, 0.5],
                         [0, 0, 0.9, 0.1],
                         [0.49, 0.01, 0.5, 0],
                         [0.49, 0.01, 0, 0.5],
                         [0.5, 0, 0.45, 0.05],
                         [0, 0.5, 0.45, 0.05],
                         [0.49, 0.01, 0.45, 0.05]]

index = 0
for r in range(2, len(offer_set) + 1):
    for subset in itertools.combinations(offer_set, r):
        binary_subset = [1 if x in subset else 0 for x in offer_set]
        p = hypothetical_choice_p[index]
        for _ in range(200):
            X.append(binary_subset)
            Y.append(generate_one_hot(p).reshape((1, len(product_set))))
        index += 1


np.random.seed(10)
X = np.array(X)
Y = np.concatenate(Y, axis=0)
dataset = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(dataset, columns=['X' + str(i) for i in product_set] + ['Y' + str(i) for i in product_set])
csv_file = 'hypothetical-4p-test.csv'
df.to_csv(csv_file, index=False)


X = []
Y = []
np.random.seed(42)
index = 0
for r in range(2, len(offer_set) + 1):
    for subset in itertools.combinations(offer_set, r):
        binary_subset = [1 if x in subset else 0 for x in offer_set]
        p = hypothetical_choice_p[index]
        for _ in range(2000):
            X.append(binary_subset)
            Y.append(generate_one_hot(p).reshape((1, len(product_set))))
        index += 1


X = np.array(X)
Y = np.concatenate(Y, axis=0)
dataset = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(dataset, columns=['X' + str(i) for i in product_set] + ['Y' + str(i) for i in product_set])
csv_file = 'hypothetical-4p-train.csv'
df.to_csv(csv_file, index=False)



"""
2. Train Test MNL Model
"""

def calc_freq(X, Y):
    X = X.float()
    Y = Y.float()
    unique_X, inverse_indices = torch.unique(X, dim=0, return_inverse=True)
    new_Y = torch.zeros_like(Y)
    print(unique_X)
    for k in range(unique_X.shape[0]):
        mask = (inverse_indices == k)
        avg_y = torch.mean(Y[mask], dim=0)
        new_Y[mask] = avg_y
        print(avg_y)
    return new_Y


class DiagonalLinear(nn.Module):
    def __init__(self, size):
        super(DiagonalLinear, self).__init__()
        self.size = size
        self.diag = nn.Parameter(torch.ones(size))

    def forward(self, x):
        # 创建对角矩阵
        diag_matrix = torch.diag(self.diag)
        return x @ diag_matrix


class MNL(nn.Module):
    def __init__(self, OptSize):
        super().__init__()
        self.input_linear = DiagonalLinear(OptSize)

    def forward(self, e):
        mask = e == 1
        e = self.input_linear(e)
        masked_e = torch.where(mask == 1, e, float('-inf'))
        masked_softmax_e = F.softmax(masked_e, dim=-1)
        return masked_softmax_e, e


def log_likelihood(out, y, safe_log=0):
    ones_indices = y == 1
    probabilities = out[ones_indices]
    negative_log_probabilities = -torch.log(probabilities + safe_log)
    total_negative_log_prob = torch.sum(negative_log_probabilities)
    return total_negative_log_prob / y.shape[0]


def train_synthetic(loss_name, num_epochs):
    input_dim = 4
    main_network = MNL(input_dim)
    train_file_path = 'hypothetical-4p-train.csv'
    test_file_path = 'hypothetical-4p-test.csv'
    df_train = pd.read_csv(train_file_path)
    df_test = pd.read_csv(test_file_path)
    X_columns = [col for col in df_train.columns if col.startswith('X')]
    Y_columns = [col for col in df_train.columns if col.startswith('Y')]
    X_train = torch.tensor(df_train[X_columns].values, dtype=torch.float)
    Y_train = torch.tensor(df_train[Y_columns].values, dtype=torch.float)
    Y_train_freq = calc_freq(X_train, Y_train)
    X_test = torch.tensor(df_test[X_columns].values, dtype=torch.float)
    Y_test = torch.tensor(df_test[Y_columns].values, dtype=torch.float)
    Y_test_freq = calc_freq(X_test, Y_test)
    optimizer = Adam(main_network.parameters(), lr=0.01, weight_decay=0)

    in_loss = None
    if loss_name == 'NLL':
        L = log_likelihood
    else:
        L = nn.MSELoss()
    for epoch in range(num_epochs):
        output, _ = main_network(X_train)
        loss = L(output, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        in_loss = loss.item()
        with torch.no_grad():
            freq_loss = L(output, Y_train_freq)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Original Loss: {loss.item() ** 0.5:.4f},'
                  f' Frequency Loss: {freq_loss.item() ** 0.5:.8f}')
    final_output, _ = main_network(X_test)
    torch.save(main_network, 'MNL_Model-hyp.pth')
    return in_loss ** 0.5, freq_loss.item() ** 0.5,\
        L(final_output, Y_test).item() ** 0.5, L(final_output, Y_test_freq).item() ** 0.5


Loss_Name = 'MSE'
epochs = 2000
in_sample_RMSE, in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq = train_synthetic(Loss_Name, epochs)
print('in_sample_RMSE, in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq')
print(in_sample_RMSE,  in_sample_RMSE_Freq, out_sample_RMSE, out_sample_RMSE_Freq)