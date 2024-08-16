import os
import sys
from tqdm import tqdm
from tqdm.notebook import tqdm
import json
from itertools import product
from utils import *
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time 
from datetime import datetime
import copy
from collections import defaultdict


import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()


# # from rpy2.rinterface_lib import openrlib
# # # Setting environment variable for R
# # with openrlib.rlock:
# #     ro.r('Sys.setenv(R_MAX_VSIZE=32000000000)')  # Adjust the number as needed
    
# from rpy2.robjects.packages import importr, isinstalled

# def ensure_r_packages(packages):
#     for package in packages:
#         if not isinstalled(package):
#             utils = importr('utils')
#             utils.install_packages(package)

# # List of R packages required by your script
# required_packages = ['rpart', 'nnet']

# # Ensure all required packages are installed
# ensure_r_packages(required_packages)





# Generate Data
def generate_and_preprocess_data(params, replication_seed, run='train'):

    # torch.manual_seed(replication_seed)
    sample_size = params['sample_size']
    device = params['device']

    # Simulate baseline covariates
    O1 = torch.randn(5, sample_size, device=device)
    Z1 = torch.randn(sample_size, device=device)
    Z2 = torch.randn(sample_size, device=device)

    if params['noiseless']:
        Z1.fill_(0)
        Z2.fill_(0)

    # Stage 1 data simulation
        
    # Input preparation
    input_stage1 = O1.t()
    
    x1, x2, x3, x4, x5 = O1[0], O1[1], O1[2], O1[3], O1[4]
    pi_10 = torch.ones(sample_size, device=device)
    pi_11 = torch.exp(0.5 - 0.5 * x3)
    pi_12 = torch.exp(0.5 * x4)
    matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

    result1 = A_sim(matrix_pi1, stage=1)
    
    if  params['use_m_propen']:
        A1, _ = result1['A'], result1['probs']
        # probs1 = M_propen(A1, O1[[2, 3]].t(), stage=1)  # multinomial logistic regression with X3, X4
        probs1 = M_propen(A1, input_stage1, stage=1)  # multinomial logistic regression with H1
    else:         
        A1, probs1 = result1['A'], result1['probs']

    A1 += 1

    g1_opt = ((x1 > -1).float() * ((x2 > -0.5).float() + (x2 > 0.5).float())) + 1
    Y1 = torch.exp(1.5 - torch.abs(1.5 * x1 + 2) * (A1 - g1_opt).pow(2)) + Z1
    
    
    # Input preparation
    input_stage2 = torch.cat([O1.t(), A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device)], dim=1)


    # Stage 2 data simulation
    pi_20 = torch.ones(sample_size, device=device)
    pi_21 = torch.exp(0.2 * Y1 - 1)
    pi_22 = torch.exp(0.5 * x4)
    matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()
    
    result2 = A_sim(matrix_pi2, stage=2)
    
    
    if  params['use_m_propen']:
        A2, _ = result2['A'], result2['probs']
        # probs2 = M_propen(A2, O1[[0, 4]].t(), stage=2)  # multinomial logistic regression with X1, X5
        probs2 = M_propen(A2, input_stage2, stage=2)  # multinomial logistic regression with H2
    else:         
        A2, probs2 = result2['A'], result2['probs']
        
    A2 += 1

    Y1_opt = torch.exp(torch.tensor(1.5, device=device)) + Z1
    g2_opt = (x3 > -1).float() * ((Y1_opt > 0.5).float() + (Y1_opt > 3).float()) + 1

    Y2 = torch.exp(1.26 - torch.abs(1.5 * x3 - 2) * (A2 - g2_opt).pow(2)) + Z2

    if run != 'test':
      # transform Y for direct search
      Y1, Y2 = transform_Y(Y1, Y2)

    # Propensity score stack
    pi_tensor_stack = torch.stack([probs1['pi_10'], probs1['pi_11'], probs1['pi_12'], probs2['pi_20'], probs2['pi_21'], probs2['pi_22']])

    # Adjusting A1 and A2 indices
    A1_indices = (A1 - 1).long().unsqueeze(0).to(device)  # Ensure A1 indices are on the correct device
    A2_indices = (A2 - 1 + 3).long().unsqueeze(0).to(device)  # Ensure A2 indices are on the correct device

    # Gathering probabilities based on actions
    P_A1_given_H1_tensor = torch.gather(pi_tensor_stack.to(device), dim=0, index=A1_indices).squeeze(0)  # Remove the added dimension after gathering
    P_A2_given_H2_tensor = torch.gather(pi_tensor_stack.to(device), dim=0, index=A2_indices).squeeze(0)  # Remove the added dimension after gathering


    # Calculate Ci tensor
    Ci = (Y1 + Y2) / (P_A1_given_H1_tensor * P_A2_given_H2_tensor)

    if run == 'test':
        return input_stage1, input_stage2, Ci, Y1, Y2, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, g1_opt, g2_opt, Z1, Z2

    # Splitting data into training and validation sets
    train_size = int(params['training_validation_prop'] * sample_size)
    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]

    # return tuple(train_tensors), tuple(val_tensors)
    return tuple(train_tensors), tuple(val_tensors), tuple([input_stage1, input_stage2, Ci, Y1, Y2, A1, A2, pi_tensor_stack, g1_opt, g2_opt])


def surr_opt(tuple_train, tuple_val, params, config_number):
    
    sample_size = params['sample_size'] 
    
    train_losses, val_losses = [], []
    best_val_loss, best_model_stage1_params, best_model_stage2_params, epoch_num_model = float('inf'), None, None, 0

    nn_stage1 = initialize_and_prepare_model(1, params, sample_size)
    nn_stage2 = initialize_and_prepare_model(2, params, sample_size)

    optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

    #  Training and Validation data
    train_data = {'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2], 'A1': tuple_train[5], 'A2': tuple_train[6]}
    val_data = {'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2], 'A1': tuple_val[5], 'A2': tuple_val[6]}


    # Training and Validation loop for both stages
    for epoch in range(params['n_epoch']):

        train_loss = process_batches(nn_stage1, nn_stage2, train_data, params, optimizer, is_train=True)
        train_losses.append(train_loss)

        val_loss = process_batches(nn_stage1, nn_stage2, val_data, params, optimizer, is_train=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            epoch_num_model = epoch
            best_val_loss = val_loss
            best_model_stage1_params = nn_stage1.state_dict()
            best_model_stage2_params = nn_stage2.state_dict()

        # Update the scheduler with the current epoch's validation loss
        update_scheduler(scheduler, params, val_loss)

    model_dir = f"models/{params['job_id']}"
    # Check if the directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define file paths for saving models
    model_path_stage1 = os.path.join(model_dir, f'best_model_stage_surr_1_{sample_size}_config_number_{config_number}.pt')
    model_path_stage2 = os.path.join(model_dir, f'best_model_stage_surr_2_{sample_size}_config_number_{config_number}.pt')
        
    # Save the models
    torch.save(best_model_stage1_params, model_path_stage1)
    torch.save(best_model_stage2_params, model_path_stage2)
    
    return (nn_stage1, nn_stage2, (train_losses, val_losses), epoch_num_model)



def DQlearning(tuple_train, tuple_val, params, config_number):
    train_input_stage1, train_input_stage2, _, train_Y1, train_Y2, train_A1, train_A2 = tuple_train
    val_input_stage1, val_input_stage2, _, val_Y1, val_Y2, val_A1, val_A2 = tuple_val


    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(params, 1)
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(params, 2)

    train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate(config_number, nn_stage2, optimizer_2, scheduler_2, 
                                                                                   train_input_stage2, train_A2, train_Y2, 
                                                                                   val_input_stage2, val_A2, val_Y2, params, 2)

    train_Y1_hat = evaluate_model_on_actions(nn_stage2, train_input_stage2, train_A2) + train_Y1
    val_Y1_hat = evaluate_model_on_actions(nn_stage2, val_input_stage2, val_A2) + val_Y1

    train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate(config_number, nn_stage1, optimizer_1, scheduler_1, 
                                                                                   train_input_stage1, train_A1, train_Y1_hat, 
                                                                                   val_input_stage1, val_A1, val_Y1_hat, params, 1)

    return (nn_stage1, nn_stage2, (train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2))




# def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):

#     # Generate and preprocess data for evaluation
#     processed_result = generate_and_preprocess_data(params_ds, replication_seed=num_replications, run='test')
#     test_input_stage1, test_input_stage2, Ci_tensor, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2  = processed_result
#     train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]


#     # Evaluation for Tao's method
#     test_input_np = test_input_stage1.cpu().numpy()
#     x1 = test_input_np[:, 0]
#     x2 = test_input_np[:, 1]
#     x3 = test_input_np[:, 2]
#     x4 = test_input_np[:, 3]
#     x5 = test_input_np[:, 4]

#     # Load the R script containing the function
#     ro.r('source("ACWL_tao.R")')

#     # Call the R function
#     results = ro.globalenv['test_ACWL'](x1, x2, x3, x4, x5, d1_star.cpu().numpy(), d2_star.cpu().numpy(), params_ds['noiseless'], config_number, params_ds['job_id'], method= "tao")
    
#     A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.float32).to(params_dql['device'])
#     A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.float32).to(params_dql['device'])

#     # Append to DataFrame
#     new_row_Tao = {
#         'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
#         'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
#         'Predicted_A1': A1_Tao.cpu().numpy().tolist(),
#         'Predicted_A2': A2_Tao.cpu().numpy().tolist()
#     }
#     df_Tao = pd.concat([df_Tao, pd.DataFrame([new_row_Tao])], ignore_index=True)

#     # print("Tao estimator: ")
#     # Calculate policy values using the Tao estimator for Tao's method
#     V_replications_M1_pred_Tao = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_Tao, A2_Tao, P_A1_g_H1, P_A2_g_H2, config_number)
    
#     # Append policy values for Tao
#     V_replications["V_replications_M1_pred"]["Tao"].append(V_replications_M1_pred_Tao)


    

#     # # Extract results
#     # select2_test = results.rx2('select2')[0]
#     # select1_test = results.rx2('select1')[0]
#     # selects_test = results.rx2('selects')[0]

#     # print(f"TEST: Select1: {select1_test}, Select2: {select2_test}, Selects: {selects_test}")

#     # # Extracting each component of the results and convert them to tensors
#     # Y1_pred_R = torch.tensor(np.array(results.rx2('R1.a1')), dtype=torch.float32)
#     # Y2_pred_R = torch.tensor(np.array(results.rx2('R2.a1')), dtype=torch.float32)

#     # Y1_stats_R = [torch.min(Y1_pred_R), torch.max(Y1_pred_R), torch.mean(Y1_pred_R)]
#     # message = f"Y1_pred_R [min, max, mean]: {Y1_stats_R}"
#     # tqdm.write(message)
#     # message = f"Y2_pred_R [min, max, mean]: [{torch.min(Y2_pred_R)}, {torch.max(Y2_pred_R)}, {torch.mean(Y2_pred_R)}]"
#     # tqdm.write(message)

#     # # torch.mean(Y1_pred + Y2_pred): 4.660262107849121
#     # message = f'torch.mean(Y1_pred_R + Y2_pred_R): {torch.mean(Y1_pred_R + Y2_pred_R)} \n'
#     # tqdm.write(message)

#     return V_replications, df_DQL, df_DS, df_Tao


def evaluate_tao(test_input_stage1, d1_star, d2_star, params_ds, config_number):

    # Convert test input from PyTorch tensor to numpy array and retrieve individual components
    test_input_np = test_input_stage1.cpu().numpy()
    x1, x2, x3, x4, x5 = test_input_np[:, 0], test_input_np[:, 1], test_input_np[:, 2], test_input_np[:, 3], test_input_np[:, 4]

    # Load the R script that contains the required function
    ro.r('source("ACWL_tao.R")')

    # Call the R function with the parameters
    results = ro.globalenv['test_ACWL'](test_input_np, x1, x2, x3, x4, x5, d1_star.cpu().numpy(), d2_star.cpu().numpy(), 
                                      params_ds['noiseless'], config_number, params_ds['job_id'], method="tao")
    


    # Extract the decisions and convert to PyTorch tensors on the specified device
    A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.float32).to(params_ds['device'])
    A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.float32).to(params_ds['device'])


    return A1_Tao, A2_Tao



def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):

    # Generate and preprocess data for evaluation
    processed_result = generate_and_preprocess_data(params_ds, replication_seed=num_replications, run='test')
    test_input_stage1, test_input_stage2, Ci_tensor, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2  = processed_result
    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]


    A1_Tao, A2_Tao = evaluate_tao(test_input_stage1, d1_star, d2_star, params_ds, config_number)

    
    # Append to DataFrame
    new_row_Tao = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1_Tao.cpu().numpy().tolist(),
        'Predicted_A2': A2_Tao.cpu().numpy().tolist()
    }
    df_Tao = pd.concat([df_Tao, pd.DataFrame([new_row_Tao])], ignore_index=True)

    # print("Tao estimator: ")
    # Calculate policy values using the Tao estimator for Tao's method
    V_replications_M1_pred_Tao = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_Tao, A2_Tao, P_A1_g_H1, P_A2_g_H2, config_number)
    
    # Append policy values for Tao
    V_replications["V_replications_M1_pred"]["Tao"].append(V_replications_M1_pred_Tao)


    # Initialize and load models for DQL
    nn_stage1_DQL = initialize_and_load_model(1, params_dql['sample_size'], params_dql, config_number)
    nn_stage2_DQL = initialize_and_load_model(2, params_dql['sample_size'], params_dql, config_number)

    # Initialize and load models for DS
    nn_stage1_DS = initialize_and_load_model(1, params_ds['sample_size'], params_ds, config_number)
    nn_stage2_DS = initialize_and_load_model(2, params_ds['sample_size'], params_ds, config_number)
    
    # Calculate test outputs for all networks in stage 1 for DQL
    A1_DQL = compute_test_outputs(nn = nn_stage1_DQL, 
                                test_input = test_input_stage1, 
                                A_tensor = A1_tensor_test, 
                                params = params_dql, 
                                is_stage1 = True)

    # Calculate test outputs for all networks in stage 2 for DQL
    A2_DQL = compute_test_outputs(nn = nn_stage2_DQL, 
                                test_input = test_input_stage2, 
                                A_tensor = A2_tensor_test, 
                                params = params_dql, 
                                is_stage1 = False)

    # Calculate test outputs for all networks in stage 1 for DS
    A1_DS = compute_test_outputs(nn = nn_stage1_DS, 
                                test_input = test_input_stage1, 
                                A_tensor = A1_tensor_test, 
                                params = params_ds, 
                                is_stage1 = True)

    # Calculate test outputs for all networks in stage 2 for DS
    A2_DS = compute_test_outputs(nn = nn_stage2_DS, 
                                test_input = test_input_stage2, 
                                A_tensor = A2_tensor_test, 
                                params = params_ds, 
                                is_stage1 = False)

    # Append to DataFrame for DQL
    new_row_DQL = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1_DQL.cpu().numpy().tolist(),
        'Predicted_A2': A2_DQL.cpu().numpy().tolist()
    }
    df_DQL = pd.concat([df_DQL, pd.DataFrame([new_row_DQL])], ignore_index=True)
    
    # Append to DataFrame for DS
    new_row_DS = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1_DS.cpu().numpy().tolist(),
        'Predicted_A2': A2_DS.cpu().numpy().tolist()
    }
    df_DS = pd.concat([df_DS, pd.DataFrame([new_row_DS])], ignore_index=True)  # Assuming df_DS is defined similarly to df_DQL



        
    # Calculate policy values using the DR estimator for DQL
    # print("DQL estimator: ")
    V_replications_M1_pred_DQL = calculate_policy_values_W_estimator(train_tensors, params_dql, A1_DQL, A2_DQL, P_A1_g_H1, P_A2_g_H2, config_number)

    # print("DS estimator: ")
    # Calculate policy values using the DR estimator for DS
    V_replications_M1_pred_DS = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_DS, A2_DS, P_A1_g_H1, P_A2_g_H2, config_number)
    
    # value fn. 
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu().item())  
    
    message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} \n\n'
    print(message)
    
    # Append policy values for DQL
    V_replications["V_replications_M1_pred"]["DQL"].append(V_replications_M1_pred_DQL)

    # Append policy values for DS
    V_replications["V_replications_M1_pred"]["DS"].append(V_replications_M1_pred_DS)


    return V_replications, df_DQL, df_DS, df_Tao




def adaptive_contrast_tao(all_data, contrast, config_number, job_id):
    train_input_stage1, train_input_stage2, train_Ci, train_Y1, train_Y2, train_A1, train_A2, pi_tensor_stack, g1_opt, g2_opt = all_data


    # Convert all tensors to CPU and then to NumPy
    A1 = train_A1.cpu().numpy()
    probs1 = pi_tensor_stack.T[:, :3].cpu().numpy()

    A2 = train_A2.cpu().numpy()
    probs2 = pi_tensor_stack.T[:, 3:].cpu().numpy()

    R1 = train_Y1.cpu().numpy()
    R2 = train_Y2.cpu().numpy()

    g1_opt = g1_opt.cpu().numpy()
    g2_opt = g2_opt.cpu().numpy()



    train_input_np = train_input_stage1.cpu().numpy()

    print("train_input_np shape: ", train_input_np.shape)

    x1 = train_input_np[:, 0]
    x2 = train_input_np[:, 1]
    x3 = train_input_np[:, 2]
    x4 = train_input_np[:, 3]
    x5 = train_input_np[:, 4]

    # Load the R script containing the function
    ro.r('source("ACWL_tao.R")')


    # Call the R function with the numpy arrays
    results = ro.globalenv['train_ACWL'](train_input_np, job_id, x1, x2, x3, x4, x5, A1, probs1, A2, probs2, R1, R2, g1_opt, g2_opt, config_number, contrast, method="tao")
    # results = ro.globalenv['train_ACWL'](train_input_np, job_id, A1, probs1, A2, probs2, R1, R2, g1_opt, g2_opt, config_number, contrast, method="tao")

    # Extract results
    select2 = results.rx2('select2')[0]
    select1 = results.rx2('select1')[0]
    selects = results.rx2('selects')[0]

    print("select2, select1, selects: ", select2, select1, selects)

    return select2, select1, selects

    


def simulations(V_replications, params, config_number):

    columns = ['Behavioral_A1', 'Behavioral_A2', 'Predicted_A1', 'Predicted_A2']

    # Initialize separate DataFrames for DQL and DS
    df_DQL = pd.DataFrame(columns=columns)
    df_DS = pd.DataFrame(columns=columns)
    df_Tao = pd.DataFrame(columns=columns)

    losses_dict = {'DQL': {}, 'DS': {}} 
    epoch_num_model_lst = []
    
    # Clone the original config for DQlearning and surr_opt
    params_DQL = copy.deepcopy(params)
    params_DS = copy.deepcopy(params)
    
    params_DS['f_model'] = 'surr_opt'
    params_DQL['f_model'] = 'DQlearning'
    params_DQL['input_dim_stage1'] = params['input_dim_stage1'] + 1 # 5 + 1 = 6 # (H_1, A_1)
    params_DQL['input_dim_stage2'] = params['input_dim_stage2'] + 1 # 7 + 1 = 8 # (H_2, A_2)
    params_DQL['num_networks'] = 1  


    for replication in tqdm(range(params['num_replications']), desc="Replications_M1"):
        print(f"Replication # -------------->>>>>  {replication+1}")

        # Generate and preprocess data for training
        tuple_train, tuple_val, adapC_tao_Data = generate_and_preprocess_data(params, replication_seed=replication, run='train')

        # Estimate treatment regime : model --> surr_opt
        print("Training started!")
        
        # Run both models on the same tuple of data
        (select2, select1, selects) = adaptive_contrast_tao(adapC_tao_Data, params["contrast"], config_number, params["job_id"])

        # Run both models on the same tuple of data
        nn_stage1_DQL, nn_stage2_DQL, trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL, config_number)
        nn_stage1_DS, nn_stage2_DS, trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS, config_number)
        # Append epoch model results from surr_opt
        epoch_num_model_lst.append(epoch_num_model_DS)
        
        # Store losses in their respective dictionaries
        losses_dict['DQL'][replication] = trn_val_loss_tpl_DQL
        losses_dict['DS'][replication] = trn_val_loss_tpl_DS
        
        # eval_DTR
        print("Evaluation started")
        V_replications, df_DQL, df_DS, df_Tao = eval_DTR(V_replications, replication, 
                                                 nn_stage1_DQL, nn_stage2_DQL, 
                                                 nn_stage1_DS, nn_stage2_DS, 
                                                 df_DQL, df_DS, df_Tao,
                                                 params_DQL, params_DS, config_number)
                
    return V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst


def run_training(config, config_updates, V_replications, config_number, replication_seed):
    torch.manual_seed(replication_seed)
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = simulations(V_replications, local_config, config_number)
    
    if not any(V_replications[key] for key in V_replications):
        warnings.warn("V_replications is empty. Skipping accuracy calculation.")
    else:
        VF_df_DQL, VF_df_DS, VF_df_Tao = extract_value_functions_separate(V_replications)
        return VF_df_DQL, VF_df_DS, VF_df_Tao, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst
    
 
    
# parallelized 

def run_training_with_params(params):

    config, current_config, V_replications, i, config_number = params
    return run_training(config, current_config, V_replications, config_number, replication_seed=i)
 

def run_grid_search(config, param_grid):
    # Initialize for storing results and performance metrics
    results = {}
    # Initialize separate cumulative DataFrames for DQL and DS
    all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
    all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run
    all_dfs_Tao = pd.DataFrame()   # DataFrames from each Tao run

    all_losses_dicts = []  # Losses from each run
    all_epoch_num_lists = []  # Epoch numbers from each run 
    grid_replications = 1

    # Collect all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), param)) for param in product(*param_grid.values())]

    num_workers = 8 # multiprocessing.cpu_count()
    print(f'{num_workers} available workers for ProcessPoolExecutor.')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_params = {}
        #for current_config in param_combinations:
        for config_number, current_config in enumerate(param_combinations):
            for i in range(grid_replications): 
                V_replications = {
                    "V_replications_M1_pred": defaultdict(list),
                    "V_replications_M1_behavioral": [],
                }
                params = (config, current_config, V_replications, i, config_number)
                future = executor.submit(run_training_with_params, params)
                future_to_params[future] = (current_config, i)

        for future in concurrent.futures.as_completed(future_to_params):
            current_config, i = future_to_params[future]            
            performance_DQL, performance_DS, performance_Tao, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = future.result()
            
            print(f'Configuration {current_config}, replication {i} completed successfully.')
            
            # Processing performance DataFrame for both methods
            performances_DQL = pd.DataFrame()
            performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

            performances_DS = pd.DataFrame()
            performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            performances_Tao = pd.DataFrame()
            performances_Tao = pd.concat([performances_Tao, performance_Tao], axis=0)

            # Update the cumulative DataFrame for DQL with the current DataFrame results
            all_dfs_DQL = pd.concat([all_dfs_DQL, df_DQL], axis=0, ignore_index=True)

            # Update the cumulative DataFrame for DS with the current DataFrame results
            all_dfs_DS = pd.concat([all_dfs_DS, df_DS], axis=0, ignore_index=True)

            # Update the cumulative DataFrame for DS with the current DataFrame results
            all_dfs_Tao = pd.concat([all_dfs_Tao, df_Tao], axis=0, ignore_index=True)

            all_losses_dicts.append(losses_dict)
            all_epoch_num_lists.append(epoch_num_model_lst)
            
            
            # Store and log average performance across replications for each configuration
            config_key = json.dumps(current_config, sort_keys=True)

            # performances is a DataFrame with columns 'DQL' and 'DS'
            performance_DQL_mean = performances_DQL["Method's Value fn."].mean()
            performance_DS_mean = performances_DS["Method's Value fn."].mean()
            performance_Tao_mean = performances_Tao["Method's Value fn."].mean()

            behavioral_DQL_mean = performances_DQL["Behavioral Value fn."].mean()  
            behavioral_DS_mean = performances_DS["Behavioral Value fn."].mean()
            behavioral_Tao_mean = performances_Tao["Behavioral Value fn."].mean()

            # Calculating the standard deviation for "Method's Value fn."
            performance_DQL_std = performances_DQL["Method's Value fn."].std()
            performance_DS_std = performances_DS["Method's Value fn."].std()
            performance_Tao_std = performances_Tao["Method's Value fn."].std()

            # Check if the configuration key exists in the results dictionary
            if config_key not in results:
                # If not, initialize it with dictionaries for each model containing the mean values
                results[config_key] = {
                    'DQL': {"Method's Value fn.": performance_DQL_mean, 
                            "Method's Value fn. SD": performance_DQL_std, 
                            'Behavioral Value fn.': behavioral_DQL_mean},
                    'DS': {"Method's Value fn.": performance_DS_mean, 
                           "Method's Value fn. SD": performance_DS_std,
                           'Behavioral Value fn.': behavioral_DS_mean},
                    'Tao': {"Method's Value fn.": performance_Tao_mean, 
                           "Method's Value fn. SD": performance_Tao_std,
                           'Behavioral Value fn.': behavioral_Tao_mean}                
                }
            else:
                # Update existing entries with new means
                results[config_key]['DQL'].update({
                    "Method's Value fn.": performance_DQL_mean,                                 
                    "Method's Value fn. SD": performance_DQL_std, 
                    'Behavioral Value fn.': behavioral_DQL_mean
                })
                results[config_key]['DS'].update({
                    "Method's Value fn.": performance_DS_mean,
                    "Method's Value fn. SD": performance_DS_std,
                    'Behavioral Value fn.': behavioral_DS_mean
                })
                results[config_key]['DS'].update({
                    "Method's Value fn.": performance_Tao_mean, 
                    "Method's Value fn. SD": performance_Tao_std,
                    'Behavioral Value fn.': behavioral_Tao_mean
                })            

            print("Performances for configuration: %s", config_key)
            print("performance_DQL_mean: %s", performance_DQL_mean)
            print("performance_DS_mean: %s", performance_DS_mean)
            print("performance_Tao_mean: %s", performance_Tao_mean)
            print("\n\n")
        

        
    folder = f"data/{config['job_id']}"
    save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder)
    load_and_process_data(config, folder)

        
        
        
        
        

# # Sequential version  

# def run_grid_search(config, param_grid):
#     # Initialize for storing results and performance metrics
#     results = {}
#     all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
#     all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run
    
#     all_losses_dicts = []  # Losses from each run
#     all_epoch_num_lists = []  # Epoch numbers from each run 
#     grid_replications = 1

#     for params in product(*param_grid.values()):
#         current_config = dict(zip(param_grid.keys(), params))
#         performances = pd.DataFrame()

#         for i in range(grid_replications): 
#             V_replications = {
#                     "V_replications_M1_pred": defaultdict(list),
#                     "V_replications_M1_behavioral": [],
#                 }
#             #performance, df, losses_dict, epoch_num_model_lst = run_training(config, current_config, V_replications, replication_seed=i)
#             performance_DQL, performance_DS, df_DQL, df_DS, losses_dict, epoch_num_model_lst = run_training(config, current_config, 
#                                                                                                             V_replications, replication_seed=i)

#             #performances = pd.concat([performances, performance], axis=0)
#             # Processing performance DataFrame for both methods
#             performances_DQL = pd.DataFrame()
#             performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

#             performances_DS = pd.DataFrame()
#             performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            
#             #all_dfs = pd.concat([all_dfs, df], axis=0)
#             # Update the cumulative DataFrame for DQL with the current DataFrame results
#             all_dfs_DQL = pd.concat([all_dfs_DQL, df_DQL], axis=0, ignore_index=True)

#             # Update the cumulative DataFrame for DS with the current DataFrame results
#             all_dfs_DS = pd.concat([all_dfs_DS, df_DS], axis=0, ignore_index=True)

                
#             all_losses_dicts.append(losses_dict)
#             all_epoch_num_lists.append(epoch_num_model_lst)
            
               
                

#         # Store and log average performance across replications for each configuration
#         config_key = json.dumps(current_config, sort_keys=True)
        
#         # This assumes performances is a DataFrame with columns 'DQL' and 'DS'
#         performance_DQL_mean = performances_DQL["Method's Value fn."].mean()
#         performance_DS_mean = performances_DS["Method's Value fn."].mean()
        
#         behavioral_DQL_mean = performances_DQL["Behavioral Value fn."].mean()  # Assuming similar structure
#         behavioral_DS_mean = performances_DS["Behavioral Value fn."].mean()

#         # Check if the configuration key exists in the results dictionary
#         if config_key not in results:
#             # If not, initialize it with dictionaries for each model containing the mean values
#             results[config_key] = {
#                 'DQL': {"Method's Value fn.": performance_DQL_mean, 'Behavioral Value fn.': behavioral_DQL_mean},
#                 'DS': {"Method's Value fn.": performance_DS_mean, 'Behavioral Value fn.': behavioral_DS_mean}
#             }
#         else:
#             # Update existing entries with new means
#             results[config_key]['DQL'].update({
#                 "Method's Value fn.": performance_DQL_mean,
#                 'Behavioral Value fn.': behavioral_DQL_mean
#             })
#             results[config_key]['DS'].update({
#                 "Method's Value fn.": performance_DS_mean,
#                 'Behavioral Value fn.': behavioral_DS_mean
#             })
                
#         print("Performances for configuration: %s", config_key)
#         print("performance_DQL_mean: %s", performance_DQL_mean)
#         print("performance_DS_mean: %s", performance_DS_mean)
#         print("\n\n")
     

#     folder = f"data/{config['job_id']}"
#     save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder)
#     load_and_process_data(config, folder)
    
    
    
    
    
        
def main():

    # Load configuration and set up the device
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    
    
    # Get the SLURM_JOB_ID from environment variables
    job_id = os.getenv('SLURM_JOB_ID')

    # If job_id is None, set it to the current date and time formatted as a string
    if job_id is None:
        job_id = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YYYYMMDDHHMMSS
    
    config['job_id'] = job_id
    
    print("Job ID: ", job_id)

    training_validation_prop = config['training_validation_prop']
    train_size = int(training_validation_prop * config['sample_size'])
    print("Training size: %d", train_size)    
    
    
    # Define parameter grid for grid search
    param_grid = {
        'activation_function': ['elu'],
        'batch_size': [7000, 8000],
        'learning_rate': [0.07],
        'num_layers': [2],
#         'noiseless': [True, False]
    }
    # Perform operations whose output should go to the file
    run_grid_search(config, param_grid)
    

class FlushFile:
    """File-like wrapper that flushes on every write."""
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()  # Flush output after write

    def flush(self):
        self.f.flush()


# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     print(f'Total time taken: {end_time - start_time:.2f} seconds')

    
   
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Record the start time
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f'Start time: {start_time_str}')
    
    sys.stdout = FlushFile(sys.stdout)
    main()
    
    # Record the end time
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f'End time: {end_time_str}')
    
    # Calculate and log the total time taken
    total_time = end_time - start_time
    print(f'Total time taken: {total_time:.2f} seconds')


    
