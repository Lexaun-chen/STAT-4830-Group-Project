search_info = ['srch_id']
continuous_feature = ['position', 'prop_starrating', 'prop_location_score1',
                      'prop_log_historical_price', 'srch_booking_window',
                      'srch_length_of_stay', 'srch_adults_count',
                      'srch_children_count', 'srch_room_count', 'price_usd']

discrete_feature = ['prop_brand_bool', 'promotion_flag',
           'srch_saturday_night_bool', 'random_bool', 'booking_bool']


tr_data = pd.read_csv('/content/drive/MyDrive/Choice Model/train_28-32_10000.csv')
te_data = pd.read_csv('/content/drive/MyDrive/Choice Model/test_28-32_1000.csv')

# Normalize data for more stable training
scaler = MinMaxScaler()
scaler.fit(tr_data[continuous_feature])
tr_data[continuous_feature] = scaler.transform(tr_data[continuous_feature])
te_data[continuous_feature] = scaler.transform(te_data[continuous_feature])

tr_data = tr_data[search_info + continuous_feature + discrete_feature]
te_data = te_data[search_info + continuous_feature + discrete_feature]
tr_offerset, tr_sell, tr_mask = get_offer_data(tr_data.groupby('srch_id'))
te_offerset_list, te_sell_list, te_mask_list = get_offer_data(te_data.groupby('srch_id'))

def train_frank_wolfe(type_num, tr_offerset_list, tr_sell_list, tr_mask_list, opt='adam'):
    problem = Problem_FrankWolfe(tr_offerset_list, tr_sell_list, tr_mask_list)
    tl, vl = problem.initialize(val_offerset, val_sell, val_mask, patience=5)
    n = type_num
    NLL_LOSS_LIST = [problem.NLL_main.item()]
    EVALUATION_LOSS_LIST = []
    TEST_LOSS_LIST = []
    with torch.no_grad():
        f = torch.zeros(len(val_sell), problem.sales.num_products, dtype=torch.float32, device=device)  # 使用验证集数据
        for proportion, fw_func in zip(problem.proportion, problem.taste_list):
            fw = torch.exp(fw_func(torch.tensor(val_offerset, dtype=torch.float32, device=device), torch.tensor(val_mask, dtype=torch.float32, device=device)))
            f += proportion * fw
        f_log = torch.log(f)
        evaluation_loss = nn.NLLLoss()(f_log, torch.tensor(val_sell, dtype=torch.int64, device=device).argmax(dim=1))  # 计算验证集损失值
        EVALUATION_LOSS_LIST.append(evaluation_loss.item())  # 将损失值添加到列表中

        f = torch.zeros(len(te_sell_list), problem.sales.num_products, dtype=torch.float32, device=device)
        for proportion, fw_func in zip(problem.proportion, problem.taste_list):
            fw = torch.exp(fw_func(torch.tensor(te_offerset_list, dtype=torch.float32, device=device), torch.tensor(te_mask_list, dtype=torch.float32, device=device)))
            f += proportion * fw
        f_log = torch.log(f)
        test_loss = nn.NLLLoss()(f_log, torch.tensor(te_sell_list, dtype=torch.int64, device=device).argmax(dim=1))
        TEST_LOSS_LIST.append(test_loss.item())

        # print('Evaluation loss: ', evaluation_loss.item())  # 打印损失值

    for m in range(n):
        print('-----Consumer Type ' + str(m + 1) + ' Start Searching------')
        if opt == 'adam':
            problem.support_finding()
            problem.proportion_update()
        elif opt == 'lbfgs':
            problem.support_finding_lbfgs()
            problem.proportion_update()
        print('main problem loss: ', problem.NLL_main.item())
        print('-----One Iteration Done-----')
        NLL_LOSS_LIST.append(problem.NLL_main.item())
            # 在 proportion_update() 之后进行评估
        with torch.no_grad():
            f = torch.zeros(len(val_sell), problem.sales.num_products, dtype=torch.float32, device=device)  # 使用验证集数据
            for proportion, fw_func in zip(problem.proportion, problem.taste_list):
                fw = torch.exp(fw_func(torch.tensor(val_offerset, dtype=torch.float32, device=device), torch.tensor(val_mask, dtype=torch.float32, device=device)))
                f += proportion * fw
            f_log = torch.log(f)
            evaluation_loss = nn.NLLLoss()(f_log, torch.tensor(val_sell, dtype=torch.int64, device=device).argmax(dim=1))  # 计算验证集损失值
            EVALUATION_LOSS_LIST.append(evaluation_loss.item())  # 将损失值添加到列表中
            print('Evaluation loss: ', evaluation_loss.item())  # 打印损失值

            f = torch.zeros(len(te_sell_list), problem.sales.num_products, dtype=torch.float32, device=device)
            for proportion, fw_func in zip(problem.proportion, problem.taste_list):
                fw = torch.exp(fw_func(torch.tensor(te_offerset_list, dtype=torch.float32, device=device), torch.tensor(te_mask_list, dtype=torch.float32, device=device)))
                f += proportion * fw
            f_log = torch.log(f)
            test_loss = nn.NLLLoss()(f_log, torch.tensor(te_sell_list, dtype=torch.int64, device=device).argmax(dim=1))
            TEST_LOSS_LIST.append(test_loss.item())

    return problem, NLL_LOSS_LIST, EVALUATION_LOSS_LIST, TEST_LOSS_LIST, tl, vl # 返回问题、训练集损失列表和测试集损失列表


# def train_frank_wolfe(type_num, tr_offerset_list, tr_sell_list, tr_mask_list, opt='adam', loss_tolerance=1e-4, max_re_search=3):
#     problem = Problem_FrankWolfe(tr_offerset_list, tr_sell_list, tr_mask_list)
#     problem.initialize()
#     n = type_num
#     NLL_LOSS_LIST = [problem.NLL_main.item()]

#     for m in range(n):
#         print('-----Consumer Type ' + str(m + 1) + ' Start Searching------')

#         last_proportion = problem.proportion[:]  # Save current proportions
#         re_search_count = 0  # Initialize re-search counter

#         while True:  # Loop for re-search attempts

#             if opt == 'adam':
#                 problem.support_finding()
#                 problem.proportion_update()
#             elif opt == 'lbfgs':
#                 problem.support_finding_lbfgs()
#                 problem.proportion_update_lbfgs()
#             else:
#                 raise ValueError("Invalid optimizer choice. Choose 'adam' or 'lbfgs'.")

#             # Check for loss improvement
#             current_loss = problem.NLL_main.item()
#             if m > 0 and abs(current_loss - NLL_LOSS_LIST[-1]) < loss_tolerance:
#                 print("Loss not improving significantly. Deleting last support and searching again.")
#                 problem.taste_list.pop()
#                 problem.fw_list.pop()
#                 problem.proportion = last_proportion  # Restore previous proportions
#                 problem.main_problem_loss()

#                 re_search_count += 1  # Increment re-search counter
#                 if re_search_count >= max_re_search:
#                     print(f"Reached maximum re-search limit ({max_re_search}). Moving to next consumer type.")
#                     break  # Exit re-search loop
#             else:
#                 break  # Exit re-search loop if loss improved

#         print('main problem loss: ', current_loss)
#         print('-----One Iteration Done-----')
#         NLL_LOSS_LIST.append(current_loss)
