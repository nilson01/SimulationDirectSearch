import os
import sys
from tqdm import tqdm
from tqdm.notebook import tqdm
import logging
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


class FlushFile:
    """File-like wrapper that flushes on every write."""
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()  # Flush output after write

    def flush(self):
        self.f.flush()
        

        
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
    x1, x2, x3, x4, x5 = O1[0], O1[1], O1[2], O1[3], O1[4]
    pi_10 = torch.ones(sample_size, device=device)
    pi_11 = torch.exp(0.5 - 0.5 * x3)
    pi_12 = torch.exp(0.5 * x4)
    matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

    result1 = A_sim(matrix_pi1, stage=1)
    
#     A1, probs1 = result1['A'], result1['probs']

    A1, _ = result1['A'], result1['probs']
    probs1 = M_propen(A1, O1[[2, 3]].t(), stage=1)  # multinomial logistic regression with X3, X4
        
    A1 += 1

    g1_opt = ((x1 > -1).float() * ((x2 > -0.5).float() + (x2 > 0.5).float())) + 1
    Y1 = torch.exp(1.5 - torch.abs(1.5 * x1 + 2) * (A1 - g1_opt).pow(2)) + Z1

    # Stage 2 data simulation
    pi_20 = torch.ones(sample_size, device=device)
    pi_21 = torch.exp(0.2 * Y1 - 1)
    pi_22 = torch.exp(0.5 * x4)
    matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

    result2 = A_sim(matrix_pi2, stage=2)
    
#     A2, probs2 = result2['A'], result2['probs']

    A2, _ = result2['A'], result2['probs']
    probs2 = M_propen(A2, O1[[0, 4]].t(), stage=2)  # multinomial logistic regression with X1, X5
        
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
    A1_indices = (A1 - 1).long().unsqueeze(0)  # A1 actions, Subtract 1 to match index values (0, 1, 2)
    A2_indices = (A2 - 1 + 3).long().unsqueeze(0)   # A2 actions, Add +3 to match index values (3, 4, 5) for A2, with added dimension

    # Gathering probabilities based on actions
    P_A1_given_H1_tensor = torch.gather(pi_tensor_stack, dim=0, index=A1_indices).squeeze(0)  # Remove the added dimension after gathering
    P_A2_given_H2_tensor = torch.gather(pi_tensor_stack, dim=0, index=A2_indices).squeeze(0)  # Remove the added dimension after gathering

    # Calculate Ci tensor
    Ci = (Y1 + Y2) / (P_A1_given_H1_tensor * P_A2_given_H2_tensor)

    # Input preparation
    input_stage1 = O1.t()
    input_stage2 = torch.cat([O1.t(), A1.unsqueeze(1), Y1.unsqueeze(1)], dim=1)

    if run == 'test':
        return input_stage1, input_stage2, Ci, Y1, Y2, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, g1_opt, g2_opt, Z1, Z2

    # Splitting data into training and validation sets
    train_size = int(params['training_validation_prop'] * sample_size)
    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]

    return tuple(train_tensors), tuple(val_tensors)



def surr_opt(tuple_train, tuple_val, params):
    
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
    model_path_stage1 = os.path.join(model_dir, f'best_model_stage_surr_1_{sample_size}.pt')
    model_path_stage2 = os.path.join(model_dir, f'best_model_stage_surr_2_{sample_size}.pt')
    
    # Save the models
    torch.save(best_model_stage1_params, model_path_stage1)
    torch.save(best_model_stage2_params, model_path_stage2)
    
    return (nn_stage1, nn_stage2, (train_losses, val_losses), epoch_num_model)



def DQlearning(tuple_train, tuple_val, params):
    train_input_stage1, train_input_stage2, _, train_Y1, train_Y2, train_A1, train_A2 = tuple_train
    val_input_stage1, val_input_stage2, _, val_Y1, val_Y2, val_A1, val_A2 = tuple_val


    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(params, 1)
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(params, 2)

    train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate(nn_stage2, optimizer_2, scheduler_2, train_input_stage2, train_A2, train_Y2, 
                                                                                   val_input_stage2, val_A2, val_Y2, params, 2)

    train_Y1_hat = evaluate_model_on_actions(nn_stage2, train_input_stage2, train_A2) + train_Y1
    val_Y1_hat = evaluate_model_on_actions(nn_stage2, val_input_stage2, val_A2) + val_Y1

    train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate(nn_stage1, optimizer_1, scheduler_1, train_input_stage1, train_A1, train_Y1_hat, 
                                                                                   val_input_stage1, val_A1, val_Y1_hat, params, 1)

    return (nn_stage1, nn_stage2, (train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2))



def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, params_dql, params_ds):

    # Generate and preprocess data for evaluation
    processed_result = generate_and_preprocess_data(params_ds, replication_seed=num_replications, run='test')
    test_input_stage1, test_input_stage2, Ci_tensor, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2  = processed_result
    

    # Initialize and load models for DQL
    nn_stage1_DQL = initialize_and_load_model(1, params_dql['sample_size'], params_dql)
    nn_stage2_DQL = initialize_and_load_model(2, params_dql['sample_size'], params_dql)

    # Initialize and load models for DS
    nn_stage1_DS = initialize_and_load_model(1, params_ds['sample_size'], params_ds)
    nn_stage2_DS = initialize_and_load_model(2, params_ds['sample_size'], params_ds)
    
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
    df_DS = pd.concat([df_DS, pd.DataFrame([new_row_DS])], ignore_index=True)


    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]


        
    # Calculate policy values using the W estimator for DQL
    V_replications_M1_pred_DQL = calculate_policy_values_W_estimator(train_tensors, params_dql, A1_DQL, A2_DQL, P_A1_g_H1, P_A2_g_H2)

    # Calculate policy values using the W estimator for DS
    V_replications_M1_pred_DS = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_DS, A2_DS, P_A1_g_H1, P_A2_g_H2)
    
    # value fn. 
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu().item())  
    
    message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} \n\n'
    logging.info(message)
    
    # Append policy values for DQL
    V_replications["V_replications_M1_pred"]["DQL"].append(V_replications_M1_pred_DQL)

    # Append policy values for DS
    V_replications["V_replications_M1_pred"]["DS"].append(V_replications_M1_pred_DS)


    return V_replications, df_DQL, df_DS




def simulations(num_replications, V_replications, params):

    columns = ['Behavioral_A1', 'Behavioral_A2', 'Predicted_A1', 'Predicted_A2']

    # Initialize separate DataFrames for DQL and DS
    df_DQL = pd.DataFrame(columns=columns)
    df_DS = pd.DataFrame(columns=columns)

    losses_dict = {'DQL': {}, 'DS': {}} 
    epoch_num_model_lst = []
    
    # Clone the original config for DQlearning and surr_opt
    params_DQL = copy.deepcopy(params)
    params_DS = copy.deepcopy(params)
    
    params_DS['f_model'] = 'surr_opt'
    params_DQL['f_model'] = 'DQlearning'
    params_DQL['input_dim_stage1'] = 6
    params_DQL['input_dim_stage2'] = 8
    params_DQL['num_networks'] = 1  


    for replication in tqdm(range(num_replications), desc="Replications_M1"):
        logging.info(f"Replication # -------------->>>>>  {replication+1}")

        # Generate and preprocess data for training
        tuple_train, tuple_val = generate_and_preprocess_data(params, replication_seed=replication, run='train')

        # Estimate treatment regime : model --> surr_opt
        logging.info("Training started!")
        
        # Run both models on the same tuple of data
        nn_stage1_DQL, nn_stage2_DQL, trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL)
        nn_stage1_DS, nn_stage2_DS, trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS)
        # Append epoch model results from surr_opt
        epoch_num_model_lst.append(epoch_num_model_DS)
        
        # Store losses in their respective dictionaries
        losses_dict['DQL'][replication] = trn_val_loss_tpl_DQL
        losses_dict['DS'][replication] = trn_val_loss_tpl_DS
        
        
        # eval_DTR
        logging.info("Evaluation started")
        V_replications, df_DQL, df_DS = eval_DTR(V_replications, replication, 
                                                 nn_stage1_DQL, nn_stage2_DQL, 
                                                 nn_stage1_DS, nn_stage2_DS, 
                                                 df_DQL, df_DS, 
                                                 params_DQL, params_DS)
                
    return V_replications, df_DQL, df_DS, losses_dict, epoch_num_model_lst


def run_training(config, config_updates, V_replications, replication_seed):
    torch.manual_seed(replication_seed)
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df_DQL, df_DS, losses_dict, epoch_num_model_lst = simulations(local_config['num_replications'], V_replications, local_config)
    
    if not any(V_replications[key] for key in V_replications):
        logger.warning("V_replications is empty. Skipping accuracy calculation.")
    else:
        VF_df_DQL, VF_df_DS = extract_value_functions_separate(V_replications)
        return VF_df_DQL, VF_df_DS, df_DQL, df_DS, losses_dict, epoch_num_model_lst
    
  
     
        
# parallelized 
    
def run_training_with_params(params):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) 

    config, current_config, V_replications, i = params
    return run_training(config, current_config, V_replications, replication_seed=i)
 
    

def run_grid_search(config, param_grid):
    # Initialize for storing results and performance metrics
    results = {}
    # Initialize separate cumulative DataFrames for DQL and DS
    all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
    all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run

    all_losses_dicts = []  # Losses from each run
    all_epoch_num_lists = []  # Epoch numbers from each run 
    grid_replications = 1

    # Collect all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), param)) for param in product(*param_grid.values())]

    num_workers = multiprocessing.cpu_count()
    logging.info(f'{num_workers} available workers for ProcessPoolExecutor.')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_params = {}
        for current_config in param_combinations:
            for i in range(grid_replications): 
                V_replications = {
                    "V_replications_M1_pred": defaultdict(list),
                    "V_replications_M1_behavioral": [],
                }
                params = (config, current_config, V_replications, i)
                future = executor.submit(run_training_with_params, params)
                future_to_params[future] = (current_config, i)

        for future in concurrent.futures.as_completed(future_to_params):
            current_config, i = future_to_params[future]
            try:
                # performance, df, losses_dict, epoch_num_model_lst = future.result()
                
                performance_DQL, performance_DS, df_DQL, df_DS, losses_dict, epoch_num_model_lst = future.result()
                
                
                logging.info(f'Configuration {current_config}, replication {i} completed successfully.')
                

                # Processing performance DataFrame for both methods
                performances_DQL = pd.DataFrame()
                performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

                performances_DS = pd.DataFrame()
                performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

                # Update the cumulative DataFrame for DQL with the current DataFrame results
                all_dfs_DQL = pd.concat([all_dfs_DQL, df_DQL], axis=0, ignore_index=True)

                # Update the cumulative DataFrame for DS with the current DataFrame results
                all_dfs_DS = pd.concat([all_dfs_DS, df_DS], axis=0, ignore_index=True)

                all_losses_dicts.append(losses_dict)
                all_epoch_num_lists.append(epoch_num_model_lst)
                
            except Exception as exc:
                logging.error(f'Generated an exception for config {current_config}, replication {i}: {exc}')


        # Store and log average performance across replications for each configuration
        config_key = json.dumps(current_config, sort_keys=True)
        
        # This assumes performances is a DataFrame with columns 'DQL' and 'DS'
        performance_DQL_mean = performances_DQL["Method's Value fn."].mean()
        performance_DS_mean = performances_DS["Method's Value fn."].mean()
        
        behavioral_DQL_mean = performances_DQL["Behavioral Value fn."].mean()  # Assuming similar structure
        behavioral_DS_mean = performances_DS["Behavioral Value fn."].mean()

        # Check if the configuration key exists in the results dictionary
        if config_key not in results:
            # If not, initialize it with dictionaries for each model containing the mean values
            results[config_key] = {
                'DQL': {"Method's Value fn.": performance_DQL_mean, 'Behavioral Value fn.': behavioral_DQL_mean},
                'DS': {"Method's Value fn.": performance_DS_mean, 'Behavioral Value fn.': behavioral_DS_mean}
            }
        else:
            # Update existing entries with new means
            results[config_key]['DQL'].update({
                "Method's Value fn.": performance_DQL_mean,
                'Behavioral Value fn.': behavioral_DQL_mean
            })
            results[config_key]['DS'].update({
                "Method's Value fn.": performance_DS_mean,
                'Behavioral Value fn.': behavioral_DS_mean
            })
                
        logging.info("Performances for configuration: %s", config_key)
        logging.info("performance_DQL_mean: %s", performance_DQL_mean)
        logging.info("performance_DS_mean: %s", performance_DS_mean)
        logging.info("\n\n")
        
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
                
#         logging.info("Performances for configuration: %s", config_key)
#         logging.info("performance_DQL_mean: %s", performance_DQL_mean)
#         logging.info("performance_DS_mean: %s", performance_DS_mean)
#         logging.info("\n\n")
     

#     folder = f"data/{config['job_id']}"
#     save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder)
#     load_and_process_data(config, folder)
    
    
    
    
    
        
def main():

    
    # setup_logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) 

    # Load configuration and set up the device
    config = load_config()
    logging.info("Model used: %s", config['f_model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    
    
    # Get the SLURM_JOB_ID from environment variables
    job_id = os.getenv('SLURM_JOB_ID')

    # If job_id is None, set it to the current date and time formatted as a string
    if job_id is None:
        job_id = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YYYYMMDDHHMMSS
    
    config['job_id'] = job_id

    training_validation_prop = config['training_validation_prop']
    train_size = int(training_validation_prop * config['sample_size'])
    logging.info("Training size: %d", train_size)    
    
    
    # Define parameter grid for grid search
    param_grid = {
        'activation_function': ['relu'],
        'batch_size': [3072],
        'learning_rate': [0.007],
        'num_layers': [4]
    }
    # Perform operations whose output should go to the file
    run_grid_search(config, param_grid)
    
    
# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     logging.info(f'Total time taken: {end_time - start_time:.2f} seconds')

    
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f'Total time taken: {end_time - start_time:.2f} seconds')


    
