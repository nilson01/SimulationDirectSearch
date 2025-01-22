import os
import sys
from tqdm import tqdm
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
from itertools import combinations


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import numpy as np
from itertools import islice


 
# Generate Data
def generate_and_preprocess_data(params, replication_seed, config_seed, run='train'):

    # Set seed for this configuration and replication
    seed_value = config_seed * 100 + replication_seed  # Ensures unique seed for each config and replication
    torch.manual_seed(seed_value)

    print()
    print(" SEED ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
    print(" SEED ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
    print("config_seed,  replication_seed, seed_value: ", config_seed,  replication_seed, seed_value)
    print(" SEED ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
    print(" SEED ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
    print()

    sample_size = params['sample_size']
    device = params['device']

    if params['setting'] == 'linear':
        print(" LINEAR DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 2, device=device)         
         
        O2 = torch.randn(sample_size, device=device)

        if params['noiseless']:
            Z1 = torch.zeros(sample_size, device=device)  # No noise, standard deviation is 0
            Z2 = torch.zeros(sample_size, device=device)
        else:
            Z1 = torch.randn(sample_size, device=device)  # Add noise, standard deviation is 1
            Z2 = torch.randn(sample_size, device=device)

        # Probability value when there are 3 treatments
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # input_stage1 = O1
        # params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  

        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Approach 1 S1
        col_names_1 = ['pi_10', 'pi_11', 'pi_12']
        probs1 = {name: matrix_pi1[:, idx] for idx, name in enumerate(col_names_1)}
        A1 = torch.randint(1, 4, (sample_size,), device=device)

        # optimal policy decisions for 'linear'
        g1_opt = torch.full((sample_size,), 3, device=device)  # optimal policy for stage 1 is to always choose action 3
        g2_opt = torch.where((1 - O2 + g1_opt + O1.sum(dim=1)) > 0, torch.tensor(3, device=device), torch.tensor(1, device=device))

        # Reward stage 1
        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)

        # Input preparation
        # input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2.unsqueeze(1)], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # 5 # 7 + 1 = 8 # (H_2)

        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()
        
        # Approach 1 S2
        col_names_2 = ['pi_20', 'pi_21', 'pi_22']
        probs2 = {name: matrix_pi2[:, idx] for idx, name in enumerate(col_names_2)}
        A2 = torch.randint(1, 4, (sample_size,), device=device)
        
        # Reward stage 2
        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params)
        # Y2 = 15 + O2 + A2 * (1 - O2 + A1 + O1.sum(dim=1)) + Z2

    elif params['setting'] == 'tao':
        print(" TAO DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Simulate baseline covariates
        # O1 = torch.randn(5, sample_size, device=device)
        O1 = torch.randn(sample_size, 5, device=device)     
        O2 = torch.tensor([], device=device) 
        # O2 = torch.randn(sample_size, 1, device=device)
        # O2 = torch.randn(sample_size, 2, device=device)

        Z1 = torch.randn(sample_size, device=device)
        Z2 = torch.randn(sample_size, device=device)

        if params['noiseless']:
            Z1.fill_(0)
            Z2.fill_(0)

        # Stage 1 data simulation
            
        # Input preparation
        input_stage1 = O1
        params['input_dim_stage1'] = input_stage1.shape[1] # (H_1)  for DS

        x1, x2, x3, x4, x5 = O1[:, 0], O1[:, 1], O1[:, 2], O1[:, 3], O1[:, 4]
        
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

        g1_opt = ((O1[:, 0] > -1).float() * ((O1[:, 1] > -0.5).float() + (O1[:, 1] > 0.5).float())) + 1
        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params) 

        # Input preparation
        input_stage2 = torch.cat([O1, A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device)], dim=1)
        params['input_dim_stage2'] = input_stage2.shape[1] # 5 # 7 + 1 = 8 # (H_2)

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

        Y1_opt =  torch.exp(torch.tensor(1.5, device=Z1.device)) + Z1
        g2_opt = (O1[:, 2] > -1).float() * ((Y1_opt > 0.5).float() + (Y1_opt > 3).float()) + 1

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params)

    elif params['setting'] == 'scheme_5':
        print(" scheme_5 DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 3, device=device) * 10 #10  # Adjusted scale
        Z1, Z2 = torch.randn(sample_size, device=device), torch.randn(sample_size, device=device)
        O2 = torch.randn(sample_size, device=device) # torch.randn(sample_size, 1, device=device)  # 

        # Probabilities for treatments, assuming uniform distribution across 3 treatments
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # Input preparation for Stage 1          
        # input_stage1 = O1
        # params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Approach 1 S1
        col_names_1 = ['pi_10', 'pi_11', 'pi_12']
        probs1 = {name: matrix_pi1[:, idx] for idx, name in enumerate(col_names_1)}
        A1 = torch.randint(1, 4, (sample_size,), device=device)

        # Computing optimal policy decisions 
        sums = O1.sum(dim=1)
        g1_opt = 3.0 * (sums > 0).float() + 1.0 * (sums < 0).float()
        g2_opt = torch.argmax(O1**2, dim=1) + 1
        
        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)

        # Input preparation for Stage 2
        # input_stage2 = torch.cat([input_stage1, A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device), O2.unsqueeze(1).to(device)], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

        # Approach 1 S2
        col_names_2 = ['pi_20', 'pi_21', 'pi_22']
        probs2 = {name: matrix_pi2[:, idx] for idx, name in enumerate(col_names_2)}
        A2 = torch.randint(1, 4, (sample_size,), device=device)
        
        # # Approach 2 S2
        # result2 = A_sim(matrix_pi2, stage=2)
        # if  params['use_m_propen']:
        #     A2, _ = result2['A'], result2['probs']
        #     probs2 = M_propen(A2, input_stage2, stage=2)  # multinomial logistic regression with H2
        # else:         
        #     A2, probs2 = result2['A'], result2['probs']
        # A2 += 1

        # Reward stage 2
        # index = torch.arange(O1.size(0), device=device)
        # f_i_O1_A2 = O1[index, A2 - 1]**2
        # Y2 = f_i_O1_A2 * params["cnst"] + O2 * params["beta"] + params["C2"] + Z2
        # Y2 = O1[torch.arange(O1.size(0), device=device), A2 - 1]**2 * params["cnst"] + O2 * params["beta"] + params["C2"] + Z2
        
        print(f"O1: {O1.dtype}, A1: {A1.dtype}, O2: {O2.dtype}, A2: {A2.dtype}, g2_opt: {g2_opt.dtype}, Z2: {Z2.dtype}")

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params)

    elif params['setting'] == 'scheme_6':

        print(" scheme_6 DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 2, device=device)
        Z1, Z2 = torch.randn(sample_size, device=device), torch.randn(sample_size, device=device)
        O2 = torch.randn(sample_size, 2, device=device)

        # Probabilities for treatments, assuming it's the same as linear case
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # Constants C1, C2 and beta
        C1, C2 = 5.0, 5.0  # Example constants
        # Compute Y1 using g(O1) and A1
        def in_C(Ot):     
            # Extract Xt1 and Xt2 from O1
            Xt1 = Ot[:, 0].float() 
            Xt2 = Ot[:, 1].float()  

            # Calculate the condition
            return Xt2 > (Xt1**2 + 5*torch.sin(5 * Xt1**2)).float()  # Apply float to ensure precision

        # Computing optimal policy decisions 
        def dt_star(Ot):
            # Ot = (Xt1, Xt2)
            return 3 * in_C(Ot).int() + 1 * (1 - in_C(Ot).int())  # in_C is 1 if true, so 3*1+1*0=3; otherwise 1     
           
        g1_opt = dt_star(O1)
        g2_opt = dt_star(O2)   

        # Input preparation for Stage 1
        # input_stage1 = O1
        # params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  

        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Approach 1 S1
        col_names_1 = ['pi_10', 'pi_11', 'pi_12']
        probs1 = {name: matrix_pi1[:, idx] for idx, name in enumerate(col_names_1)}
        A1 = torch.randint(1, 4, (sample_size,), device=device)

        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)

        # Input preparation for Stage 2         
        # input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

        # Approach 1 S2
        col_names_2 = ['pi_20', 'pi_21', 'pi_22']
        probs2 = {name: matrix_pi2[:, idx] for idx, name in enumerate(col_names_2)}
        A2 = torch.randint(1, 4, (sample_size,), device=device)

        # # Approach 2 S2
        # result2 = A_sim(matrix_pi2, stage=2)
        # if  params['use_m_propen']:
        #     A2, _ = result2['A'], result2['probs']
        #     probs2 = M_propen(A2, input_stage2, stage=2)  # multinomial logistic regression with H2
        # else:         
        #     A2, probs2 = result2['A'], result2['probs']
        # A2 += 1

        # Reward stage 2
        # Y2 = A2 * (10 * in_C(O2) - 1) + C2 + Z2                
        # Y2 = A2 * (10 * (O2[:, 1].float()  > (O2[:, 0].float() **2 + 5*torch.sin(5 * O2[:, 0].float() **2)).float() ) - 1) + params["C2"] + Z2          

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params) 


        
        # Y1_g1_opt =  calculate_reward_stage1(O1, g1_opt, g1_opt, Z1, params)
        # Y2_g2_opt = calculate_reward_stage2(O1, g1_opt, O2, g2_opt, g2_opt, Z2, params)
        # print("Y1_g1_opt mean: ", torch.mean(Y1_g1_opt) )
        # print("Y2_g2_opt mean: ", torch.mean(Y2_g2_opt) )         
        # print("Y1_g1_opt+Y2_g2_opt mean: ", torch.mean(Y1_g1_opt+Y2_g2_opt) )
  
    elif params['setting'] == 'scheme_7':

        print(" scheme_i DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 3, device=device)
        Z1, Z2 = torch.randn(sample_size, device=device), torch.randn(sample_size, device=device)
        O2 = torch.randn(sample_size, 3, device=device)

        # Probabilities for treatments, assuming it's the same as linear case
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # Input preparation for Stage 1
        # input_stage1 = O1
        # params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Approach 1 S1
        col_names_1 = ['pi_10', 'pi_11', 'pi_12']
        probs1 = {name: matrix_pi1[:, idx] for idx, name in enumerate(col_names_1)}
        A1 = torch.randint(1, 4, (sample_size,), device=device)

        # Compute Y1 using g(O1) and A1
        def in_C(Ot):     
            # Extract Xt1 and Xt2 from O1
            X = Ot[:, 0].float() 
            Y = Ot[:, 1].float()  
            Z = Ot[:, 2].float()  

            # Calculate the condition
            return Z > -1.0 + (X**2).float() + torch.cos(8*X**2+Y).float() + (Y**2).float() + 2*torch.sin(5*Y**2).float()   
        
        # Computing optimal policy decisions 
        def dt_star(Ot):
            return 3 * in_C(Ot).int() + 1 * (1 - in_C(Ot).int())  # in_C is 1 if true, so 3*1+1*0=3; otherwise 1     
           
        g1_opt = dt_star(O1)
        g2_opt = dt_star(O2)

        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)

        # Input preparation for Stage 2         
        # input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

        # Approach 1 S2
        col_names_2 = ['pi_20', 'pi_21', 'pi_22']
        probs2 = {name: matrix_pi2[:, idx] for idx, name in enumerate(col_names_2)}
        A2 = torch.randint(1, 4, (sample_size,), device=device)

        # # Approach 2 S2

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params) 

    elif params['setting'] == 'scheme_8':

        print(" scheme_8 DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 3, device=device)
        Z1, Z2 = torch.randn(sample_size, device=device), torch.randn(sample_size, device=device)
        O2 = torch.randn(sample_size, 2, device=device)

        # Probabilities for treatments, assuming it's the same as linear case
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # Constants C1, C2 and beta 
        # C1, C2 = 5.0, 5.0  # Example constants
        # Compute Y1 using g(O1) and A1
        def in_C(Ot):     
            # Extract Xt1 and Xt2 from O1
            Xt1 = Ot[:, 0].float() 
            Xt2 = Ot[:, 1].float()  

            # Calculate the condition
            return Xt2 > (Xt1**2 + 5*torch.sin(5 * Xt1**2)).float()  # Apply float to ensure precision

        # Computing optimal policy decisions 
        def dt_star(Ot):
            # Ot = (Xt1, Xt2)
            return 3 * in_C(Ot).int() + 1 * (1 - in_C(Ot).int())  # in_C is 1 if true, so 3*1+1*0=3; otherwise 1     
           
        g1_opt_orig = dt_star(O1)
        g2_opt_orig = torch.argmax(O1**2, dim=1) + 1




        # # Input preparation for Stage 1
        # input_stage1 = O1
        # if params['interaction_terms'] and O1.shape[1] > 1:
        #     interaction_terms = [O1[:, i:i+1] * O1[:, j:j+1] for i, j in combinations(range(O1.shape[1]), 2)]
        #     # Concatenate the original features and interaction terms
        #     input_stage1 = torch.cat([O1] + interaction_terms, dim=1)
            
        
        # params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Approach 1 S1
        col_names_1 = ['pi_10', 'pi_11', 'pi_12']
        probs1 = {name: matrix_pi1[:, idx] for idx, name in enumerate(col_names_1)}
        A1 = torch.randint(1, 4, (sample_size,), device=device)



        zeros_tensor = torch.ones_like(A1)
        ones_tensor = torch.full_like(A1, 2)
        twos_tensor = torch.full_like(A1, 3)

        # Now call the function with these tensors
        y1_ones = calculate_reward_stage1(O1, zeros_tensor, g1_opt_orig, Z1, params)         
        y1_twos = calculate_reward_stage1(O1, ones_tensor, g1_opt_orig, Z1, params)         
        y1_tres = calculate_reward_stage1(O1, twos_tensor, g1_opt_orig, Z1, params)

        stacked_tensors = torch.stack([y1_ones, y1_twos, y1_tres], dim=0)
        g1_opt = torch.argmax(stacked_tensors, dim=0) + 1


        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)



        # Element-wise comparison and check if all elements are the same
        comparison = torch.eq(g1_opt_orig, g1_opt)
        all_same = torch.all(comparison)

        # Debug print
        print("**********   Comparison Result:", "Exactly the same." if all_same else "Differences found.")
        if not all_same:
            mismatched_indices = torch.nonzero(~comparison, as_tuple=True)
            print("Mismatched Indices:", mismatched_indices)
            print("g1_opt_orig values:", g1_opt_orig[mismatched_indices])
            print("g1_opt values:", g1_opt[mismatched_indices])


        # # Input preparation for Stage 2  
        # input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2], dim=1)       
        # if O2.ndimension() == 1: # O2 is 1D, treated as a single column
        #     input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2.unsqueeze(1).to(device)], dim=1)       
        # # Input preparation for Stage 2
        # if params['interaction_terms']:
        #     interaction_terms = [input_stage2[:, i:i+1] * input_stage2[:, j:j+1] for i, j in combinations(range(input_stage2.shape[1]), 2)]
        #     # Concatenate the original features and interaction terms
        #     input_stage2 = torch.cat([input_stage2] + interaction_terms, dim=1)
            
        # # input_stage2 = torch.cat([O1, A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device), O2.unsqueeze(1).to(device)], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

        # Approach 1 S2
        col_names_2 = ['pi_20', 'pi_21', 'pi_22']
        probs2 = {name: matrix_pi2[:, idx] for idx, name in enumerate(col_names_2)}
        A2 = torch.randint(1, 4, (sample_size,), device=device)


        # Create tensors of all zeros, ones, and twos for A2
        zeros_tensor_A2 = torch.ones_like(A2)
        ones_tensor_A2 = torch.full_like(A2, 2)
        twos_tensor_A2 = torch.full_like(A2, 3)

        # Calculate rewards for each tensor
        y2_ones = calculate_reward_stage2(O1, g1_opt, O2, zeros_tensor_A2, g2_opt_orig, Z2, params)
        y2_twos = calculate_reward_stage2(O1, g1_opt, O2, ones_tensor_A2, g2_opt_orig, Z2, params)
        y2_tres = calculate_reward_stage2(O1, g1_opt, O2, twos_tensor_A2, g2_opt_orig, Z2, params)

        # Stack the tensors and find the argmax along the new dimension, then add 1
        g2_opt = torch.argmax(torch.stack([ y2_ones, y2_twos, y2_tres], dim=0), dim=0) + 1

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params) 


    elif params['setting'] == 'new':
        # Set mu_0 and sigma^2 for sampling θ_{ta}
        # mu_0 = 2.0
        # sigma_squared = 0.5

        # params['gamma1'] = torch.randn(params['input_dim'], device=device)
        # params['gamma2'] = torch.randn(params['input_dim'], device=device)
        # params['gamma1_prime'] = torch.randn(params['input_dim'], device=device)
        # params['gamma2_prime'] = torch.randn(params['input_dim'], device=device)

        # params.update({
        #     'gamma1': torch.randn(10, device=device),
        #     'gamma2': torch.randn(10, device=device),
        #     'delta_A1': torch.tensor([1.0, 10.0, 1.0], device=device),  # Amplify difference, Optimal action is A1=2
        #     'delta_A2': torch.tensor([1.0, 10.0, 1.0], device=device),  # Optimal action is A2=2
        #     'lambda_val': 0.3,  # Dependency on Y1 in Y2
        # })
        

        # Simulate baseline covariates
        O1 = torch.randn(sample_size, params['input_dim'], device=device)  # X ∈ R^10
        O2 = torch.tensor([], device=device)  # Not used in this scheme

        Z1 = torch.randn(sample_size, device=device) * 0.5  # ε1 ~ N(0, 0.5^2)
        Z2 = torch.randn(sample_size, device=device) * 0.5  # ε2 ~ N(0, 0.5^2)

        # Sample θ_{ta} for biased treatment assignment at stage 1
        # mu_t1 = torch.ones(params['input_dim'], device=device) * mu_0  # Mean vector for suboptimal actions
        # mu_t2 = -torch.ones(params['input_dim'], device=device) * mu_0  # Mean vector for optimal action



        # Approximate optimal action g1_opt
        # delta_A1 = params['delta_A1']
        # gamma1 = params['gamma1']
        # inner_product = torch.matmul(O1, gamma1)
        # expected_rewards = (torch.sin(inner_product).unsqueeze(1) * delta_A1)**2  # Square to match reward function
        # g1_opt = expected_rewards.argmax(dim=1) + 1  # Actions in {1,2,3}


        # Compute components for optimal policy at Stage 1
        sin_component_opt1 = torch.sin(torch.matmul(O1, params['gamma1']))
        cos_component_opt1 = torch.cos(torch.matmul(O1, params['gamma1_prime']))
        expected_rewards1 = (params['delta_A1'].unsqueeze(0) * sin_component_opt1.unsqueeze(1))**2 + \
                            (params['eta_A1'].unsqueeze(0) * cos_component_opt1.unsqueeze(1))
        g1_opt = expected_rewards1.argmax(dim=1) + 1  # Actions in {1,2,3}






        # theta_1 = torch.zeros(3, 10, device=device)
        # # For action a=1 (suboptimal)
        # theta_1[0] = mu_t1 + torch.randn(10, device=device) * np.sqrt(sigma_squared)
        # # For action a=2 (optimal)
        # theta_1[1] = mu_t2 + torch.randn(10, device=device) * np.sqrt(sigma_squared)
        # # For action a=3 (suboptimal)
        # theta_1[2] = mu_t1 + torch.randn(10, device=device) * np.sqrt(sigma_squared)

        # Sample theta_{ta} for Stage 1 

        # tuning_param = torch.tensor(1.0)
        theta_mu = torch.full((params['input_dim'],), 1, device=device)

        # theta_mu = torch.randn(params['input_dim'], device=device)  # Random vector not aligned with gamma1

        # Approach: 1 theta_1    
        theta_1 = torch.zeros(3, params['input_dim'], device=device)
        for a in range(3):
            if a + 1 != g1_opt.mode()[0].item():  # If not the most common optimal action
                theta_1[a] = -theta_mu 
            else:
                theta_1[a] = theta_mu 


        # Stage 1 data simulation
        # Compute treatment probabilities π_1(A1 = a | H1)
        # Split exp_theta_X into three separate components 
        exp_theta_X = torch.exp(torch.matmul(O1, theta_1.t()))  # Shape: (n, 3)
        pi_10, pi_11, pi_12 = exp_theta_X[:, 0], exp_theta_X[:, 1], exp_theta_X[:, 2]
        # Stack the components along the 0th dimension and transpose to match the shape (n, 3)
        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()
        result1 = A_sim(matrix_pi1, stage=1)
        A1, probs1 = result1['A'], result1['probs']

        A1 += 1  # Adjust actions to be in {1,2,3}




        # Calculate Y1
        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)     
        Y1_opt =  calculate_reward_stage1(O1, g1_opt, g1_opt, Z1, params)


        # Prepare input for stage 2
        input_stage2 = torch.cat([O1, A1.unsqueeze(1).float().to(device), Y1.unsqueeze(1).to(device)], dim=1)
        params['input_dim_stage2'] = input_stage2.shape[1]  # Update input dimension for stage 2


        # Approximate optimal action g2_opt
        # delta_A2 = params['delta_A2']
        # gamma2 = params['gamma2']
        # inner_product2 = torch.matmul(O1, gamma2)
        # # expected_rewards2 = (torch.cos(inner_product2).unsqueeze(1) * delta_A2)**2 + params['lambda_val'] * Y1.unsqueeze(1)
        # expected_rewards2 = (torch.cos(inner_product2).unsqueeze(1) * delta_A2)**2 + params['lambda_val']* Y1.unsqueeze(1)
        # g2_opt = expected_rewards2.argmax(dim=1) + 1  # Actions in {1,2,3}

        # Compute components for optimal policy at Stage 2
        cos_component_opt2 = torch.cos(torch.matmul(O1, params['gamma2']))
        sin_component_opt2 = torch.sin(torch.matmul(O1, params['gamma2_prime']))

        expected_rewards2 = (params['delta_A2'].unsqueeze(0) * cos_component_opt2.unsqueeze(1))**2 + \
                            (params['eta_A2'].unsqueeze(0) * sin_component_opt2.unsqueeze(1)) + \
                            params['lambda_val'] * Y1_opt.unsqueeze(1) 
        
        # expected_rewards2 = (params['delta_A2'].unsqueeze(0) * cos_component_opt2.unsqueeze(1))**2 + \
        #             (params['eta_A2'].unsqueeze(0) * sin_component_opt2.unsqueeze(1)) 
        g2_opt = expected_rewards2.argmax(dim=1) + 1  # Actions in {1,2,3}


 
        # Sample θ_{ta} for biased treatment assignment at stage 2
        # theta_2 = torch.zeros(3, 10, device=device)
        # # For action a=1 (suboptimal)
        # theta_2[0] = mu_t1 + torch.randn(10, device=device) * np.sqrt(sigma_squared)
        # # For action a=2 (optimal)
        # theta_2[1] = mu_t2 + torch.randn(10, device=device) * np.sqrt(sigma_squared)
        # # For action a=3 (suboptimal)
        # theta_2[2] = mu_t1 + torch.randn(10, device=device) * np.sqrt(sigma_squared)

        # Sample theta_{ta} for Stage 2
        # Approach: 1 theta_1        
        theta_2 = torch.zeros(3, params['input_dim'], device=device)
        for a in range(3):
            if a + 1 != g2_opt.mode()[0].item():  # If not the most common optimal action
                theta_2[a] = -theta_mu
            else:
                theta_2[a] = theta_mu 

        # # Approach: 2 theta_2        
        # theta_2 = torch.zeros(3, params['input_dim'], device=device) 
        # for a in range(3):
        #     if a + 1 != g2_opt.mode()[0].item():  # If not the most common optimal action
        #         theta_1[a] = -theta_mu 
        #     else:
        #         theta_1[a] = theta_mu 

        alpha = 0.5  # Dependency on A1
        beta = 0.5   # Dependency on Y1

        # Compute treatment probabilities π_2(A2 = a | H2)
        # Split exp_theta_X into three separate components 
        exp_theta_X_stage2 = torch.exp(torch.matmul(O1, theta_2.t()) + alpha * A1.unsqueeze(1) + beta * Y1.unsqueeze(1))
        pi_20, pi_21, pi_22 = exp_theta_X_stage2[:, 0], exp_theta_X_stage2[:, 1], exp_theta_X_stage2[:, 2]
        # Stack the components along the 0th dimension and transpose to match the shape (n, 3)
        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()
        result2 = A_sim(matrix_pi2, stage=2)
        A2, probs2 = result2['A'], result2['probs']

        A2 += 1  # Adjust actions to be in {1,2,3}



        # Calculate Y2
        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params, Y1=Y1) 





   
    elif params['setting'] == 'scheme_i':

        print(" scheme_i DGP setting ::::::::::------------------------------>>>>>>>>>>>>>>>>> ")
        # Generate data using PyTorch
        O1 = torch.randn(sample_size, 3, device=device)
        Z1, Z2 = torch.randn(sample_size, device=device), torch.randn(sample_size, device=device)
        O2 = torch.randn(sample_size, device=device)

        # Probabilities for treatments, assuming it's the same as linear case
        pi_value = torch.full((sample_size,), 1 / 3, device=device)
        pi_10 = pi_11 = pi_12 = pi_20 = pi_21 = pi_22 = pi_value

        # Input preparation for Stage 1
        input_stage1 = O1
        params['input_dim_stage1'] = input_stage1.shape[1] # (H_1)  for DS

        matrix_pi1 = torch.stack((pi_10, pi_11, pi_12), dim=0).t()

        # Simulating actions based on probabilities using A_sim function
        result1 = A_sim(matrix_pi1, stage=1)
        if params['use_m_propen']:
            A1, _ = result1['A'], result1['probs']
            probs1 = M_propen(A1, input_stage1, stage=1)  # Multinomial logistic regression with O1
        else:
            A1, probs1 = result1['A'], result1['probs']

        A1 += 1

        # Constants C1, C2 and beta
        C1, C2, beta = 1.0, 2.0, 0.5  # Example constants

        # Compute Y1 using g(O1) and A1
        def g(O1):
            return O1.sum(dim=1)  # Example g function
        Y1 = A1 * g(O1) + C1 + Z1

        # Input preparation for Stage 2
        input_stage2 = torch.cat([O1.t(), A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device), O2.unsqueeze(1).to(device)], dim=1)
        params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)
        matrix_pi2 = torch.stack((pi_20, pi_21, pi_22), dim=0).t()

        # Simulating actions for Stage 2
        result2 = A_sim(matrix_pi2, stage=2)
        if params['use_m_propen']:
            A2, _ = result2['A'], result2['probs']
            probs2 = M_propen(A2, input_stage2, stage=2)  # Multinomial logistic regression with combined Stage 1 and 2 inputs
        else:
            A2, probs2 = result2['A'], result2['probs']

        A2 += 1

        # Define f_i function (example assuming some functionality, adjust as needed)
        def f_i_scheme1(O1, A1, i):
            return 1.5 * A1 * O1[:, i - 1] + A1 * (O1[:, i - 1] ** 2) / 2

        # Compute Y2 using f_i and other inputs
        Y2 = sum(f_i_scheme1(O1, A1, i) * (A2 == i).float() for i in range(1, 4)) + O2 * beta + C2 + Z2


    # Input preparation for Stage 1
    input_stage1 = O1
    if params['interaction_terms'] and O1.shape[1] > 1:
        interaction_terms = [O1[:, i:i+1] * O1[:, j:j+1] for i, j in combinations(range(O1.shape[1]), 2)]
        # Concatenate the original features and interaction terms
        input_stage1 = torch.cat([O1] + interaction_terms, dim=1)
        
    params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)

    # Input preparation for Stage 2  
      
    if O2.ndimension() == 1 and O2.numel()>0: # O2 is 1D, treated as a single column
        input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2.unsqueeze(1).to(device)], dim=1)      
    else:
        input_stage2 = torch.cat([O1, A1.unsqueeze(1), Y1.unsqueeze(1), O2], dim=1)     

    # Input preparation for Stage 2
    if params['interaction_terms']:
        interaction_terms = [input_stage2[:, i:i+1] * input_stage2[:, j:j+1] for i, j in combinations(range(input_stage2.shape[1]), 2)]
        # Concatenate the original features and interaction terms
        input_stage2 = torch.cat([input_stage2] + interaction_terms, dim=1)
        
    # input_stage2 = torch.cat([O1, A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device), O2.unsqueeze(1).to(device)], dim=1)
    params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

    
    if run != 'test':
      # transform Y for direct search
      Y1, Y2 = transform_Y(Y1, Y2)

    Y1_g1_opt =  calculate_reward_stage1(O1, g1_opt, g1_opt, Z1, params)
    Y2_g2_opt = calculate_reward_stage2(O1, g1_opt, O2, g2_opt, g2_opt, Z2, params, Y1 = Y1_g1_opt)

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
        print("="*90)
        print("pi_10: ", probs1['pi_10'].mean().item(), "pi_11: ", probs1['pi_11'].mean().item(), "pi_12: ", probs1['pi_12'].mean().item())
        print("pi_20: ", probs2['pi_20'].mean().item(), "pi_21: ", probs2['pi_21'].mean().item(), "pi_22: ", probs2['pi_22'].mean().item())
        print("="*90)
        print()

        print("="*60)
        print("Y1_beh mean: ", torch.mean(Y1) )
        print("Y2_beh mean: ", torch.mean(Y2) )         
        print("Y1_beh+Y2_beh mean: ", torch.mean(Y1+Y2) )

        print()
        print("Unique- Optimal treat Stage 1", g1_opt.unique())
        print("Unique- Optimal treat Stage 2", g2_opt.unique())
        print()

        print("Y1_g1_opt mean: ", torch.mean(Y1_g1_opt) )
        print("Y2_g2_opt mean: ", torch.mean(Y2_g2_opt) )         
        print("Y1_g1_opt+Y2_g2_opt mean: ", torch.mean(Y1_g1_opt+Y2_g2_opt) )

        print("="*60)
        return input_stage1, input_stage2, O2, Y1, Y2, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, g1_opt, g2_opt, Z1, Z2, Y1_g1_opt, Y2_g2_opt

    # Splitting data into training and validation sets
    train_size = int(params['training_validation_prop'] * Y1.shape[0])
    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]

    # return tuple(train_tensors), tuple(val_tensors)
    return tuple(train_tensors), tuple(val_tensors), tuple([O1, O2, Y1, Y2, A1, A2, pi_tensor_stack, g1_opt, g2_opt])


# def surr_opt(tuple_train, tuple_val, params, config_number, ensemble_num, option_sur):
    
#     sample_size = params['sample_size'] 
    
#     train_losses, val_losses = [], []
#     best_val_loss, best_model_stage1_params, best_model_stage2_params, epoch_num_model = float('inf'), None, None, 0

#     nn_stage1 = initialize_and_prepare_model(1, params)
#     nn_stage2 = initialize_and_prepare_model(2, params)

#     optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

#     #  Training and Validation data
#     train_data = {'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2], 'A1': tuple_train[5], 'A2': tuple_train[6]}
#     val_data = {'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2], 'A1': tuple_val[5], 'A2': tuple_val[6]}


#     # Training and Validation loop for both stages
#     for epoch in range(params['n_epoch']):

#         train_loss = process_batches(nn_stage1, nn_stage2, train_data, params, optimizer, option_sur=option_sur, is_train=True)
#         train_losses.append(train_loss)

#         val_loss = process_batches(nn_stage1, nn_stage2, val_data, params, optimizer, option_sur=option_sur, is_train=False)
#         val_losses.append(val_loss)

#         if val_loss < best_val_loss:
#             epoch_num_model = epoch
#             best_val_loss = val_loss
#             best_model_stage1_params = nn_stage1.state_dict()
#             best_model_stage2_params = nn_stage2.state_dict()

#         # Update the scheduler with the current epoch's validation loss
#         update_scheduler(scheduler, params, val_loss)

#     model_dir = f"models/{params['job_id']}"
#     # Check if the directory exists, if not, create it
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
    
#     # Define file paths for saving models
#     model_path_stage1 = os.path.join(model_dir, f'best_model_stage_surr_1_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')
#     model_path_stage2 = os.path.join(model_dir, f'best_model_stage_surr_2_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')
        
#     # Save the models
#     torch.save(best_model_stage1_params, model_path_stage1)
#     torch.save(best_model_stage2_params, model_path_stage2)
    
#     return ((train_losses, val_losses), epoch_num_model)




class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = data['input1'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.data.items()}
        return sample



# def surr_opt(tuple_train, tuple_val, params, config_number, ensemble_num, option_sur):
#     sample_size = params['sample_size']
#     device = params['device']
#     n_epoch = params['n_epoch']
#     batch_size = params['batch_size']
#     eval_freq = params.get('eval_freq', 10)  # Evaluate every 10 steps by default

#     # Initialize models
#     nn_stage1 = initialize_and_prepare_model(1, params)
#     nn_stage2 = initialize_and_prepare_model(2, params)

#     # Initialize optimizer and scheduler
#     optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

#     # Prepare training and validation data
#     train_data = {
#         'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2],
#         'A1': tuple_train[5], 'A2': tuple_train[6]
#     }
#     val_data = {
#         'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2],
#         'A1': tuple_val[5], 'A2': tuple_val[6]
#     }

#     # Create datasets and data loaders
#     train_dataset = CustomDataset(train_data)
#     val_dataset = CustomDataset(val_data)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Training variables
#     train_losses = []
#     val_losses = []
#     best_val_loss = float('inf')
#     best_model_stage1_params = None
#     best_model_stage2_params = None
#     epoch_num_model = 0
#     total_steps = 0 

#     # For tabular display
#     loss_table = pd.DataFrame(columns=['Epoch', 'Avg Training Loss', 'Avg Validation Loss'])


#     for epoch in range(n_epoch):
#         nn_stage1.train()
#         nn_stage2.train()

#         # Accumulate loss over batches for the epoch
#         running_train_loss = 0.0
#         running_val_loss = 0.0
#         num_batches = 0 
#         num_val_steps = 0

        

#         for batch_data in train_loader:
#             total_steps += 1  # Total Steps means the total number of batches in an epoch           
#             num_batches += 1

#             batch_data = {k: v.to(device) for k, v in batch_data.items()}

#             # Forward pass
#             outputs_stage1 = nn_stage1(batch_data['input1'])
#             outputs_stage2 = nn_stage2(batch_data['input2'])

#             outputs_stage1 = torch.stack(outputs_stage1, dim=1).squeeze()
#             outputs_stage2 = torch.stack(outputs_stage2, dim=1).squeeze()

#             # Compute loss
#             loss = main_loss_gamma(
#                 outputs_stage1, outputs_stage2, batch_data['A1'], batch_data['A2'], 
#                         batch_data['Ci'], option=option_sur, surrogate_num=params['surrogate_num']
#                         )

#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()

#             # Gradient Clipping (to prevent exploding gradients)
#             if params['gradient_clipping']:
#                 torch.nn.utils.clip_grad_norm_(nn_stage1.parameters(), max_norm=1.0)
#                 torch.nn.utils.clip_grad_norm_(nn_stage2.parameters(), max_norm=1.0)

#             optimizer.step() 
#             running_train_loss += loss.item()

#             # Evaluate model at specified frequency
#             if total_steps % eval_freq == 0:            
#                 val_loss = evaluate_model(nn_stage1, nn_stage2, val_loader, params)
#                 running_val_loss += val_loss
#                 num_val_steps += 1

#                 # Save the best model
#                 if val_loss < best_val_loss:
#                     epoch_num_model = epoch
#                     best_val_loss = val_loss
#                     best_model_stage1_params = copy.deepcopy(nn_stage1.state_dict())
#                     best_model_stage2_params = copy.deepcopy(nn_stage2.state_dict())

#                 # Update scheduler if necessary
#                 update_scheduler(scheduler, params, val_loss)

#         # Calculate and store average training loss for this epoch
#         avg_train_loss = running_train_loss / num_batches
#         train_losses.append(avg_train_loss)  

#         # avg_val_loss = running_val_loss / num_val_steps
#         # val_losses.append(avg_val_loss)

#         if num_val_steps > 0:
#             avg_val_loss = running_val_loss / num_val_steps
#             val_losses.append(avg_val_loss)
#         else:
#             val_losses.append(float('nan'))  # In case there were no validation steps this epoch
        
#         # print(f'Epoch [{epoch+1}/{n_epoch}], Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss if num_val_steps > 0 else "N/A"}')
        
#         # Append to the table
#         # loss_table = loss_table.append({
#         #     'Epoch': epoch + 1,
#         #     'Avg Training Loss': avg_train_loss,
#         #     'Avg Validation Loss': avg_val_loss if num_val_steps > 0 else "N/A"
#         # }, ignore_index=True)

#         new_row = pd.DataFrame({
#             'Epoch': [epoch + 1],
#             'Avg Training Loss': [avg_train_loss],
#             'Avg Validation Loss': [avg_val_loss if num_val_steps > 0 else "N/A"]
#         })
#         loss_table = pd.concat([loss_table, new_row], ignore_index=True)





#     # Print the table at the end of training
#     print("LOSS TABLE: \n")
#     print(loss_table.to_string(index=False))

#     # Save the best model parameters
#     model_dir = f"models/{params['job_id']}"
#     os.makedirs(model_dir, exist_ok=True)

#     # Define file paths for saving models
#     model_path_stage1 = os.path.join(
#         model_dir, f'best_model_stage_surr_1_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')
    
#     model_path_stage2 = os.path.join(
#         model_dir, f'best_model_stage_surr_2_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')
       

#     torch.save(best_model_stage1_params, model_path_stage1)
#     torch.save(best_model_stage2_params, model_path_stage2)

#     return ((train_losses, val_losses), epoch_num_model)






# # Function to create DataLoader with limited number of batches
# def get_random_loader(dataset, batch_size, batches_to_sample = 30):
#     # Calculate the total number of samples needed for the specified number of batches
#     total_samples = batches_to_sample * batch_size
    
#     # Randomly sample the required indices from the dataset
#     indices = np.random.choice(len(dataset), total_samples, replace=True)
    
#     # Use SubsetRandomSampler to sample only these indices
#     sampler = SubsetRandomSampler(indices)
#     loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
#     return loader



# # Define a function to create the DataLoader with repeated shuffling if needed
# def get_data_loader_with_reshuffle(dataset, batch_size, batches_to_sample):
#     # Generate all indices for the dataset
#     indices = np.arange(len(dataset))
    
#     # Prepare a list to hold the selected indices for this epoch
#     sampled_indices = []
    
#     # Loop until we reach the desired number of batches
#     while len(sampled_indices) < batches_to_sample * batch_size:
#         # Shuffle indices
#         np.random.shuffle(indices)
        
#         # Add the shuffled indices to sampled_indices
#         sampled_indices.extend(indices)
    
#     # Trim sampled_indices to match exactly `num_batches * batch_size`
#     sampled_indices = sampled_indices[:batches_to_sample * batch_size]
    
#     # Create DataLoader with SubsetRandomSampler to sample these indices
#     sampler = SubsetRandomSampler(sampled_indices)
#     loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
#     return loader



def surr_opt(tuple_train, tuple_val, params, config_number, ensemble_num, option_sur, seed_value):
    sample_size = params['sample_size']
    device = params['device']
    n_epoch = params['n_epoch']
    batch_size = params['batch_size']
    # batches_to_sample =  params['batches_to_sample']
    
    eval_freq =  params['eval_freq'] # params.get('eval_freq', 10)  # Evaluate every 10 steps by default
    # early_stopping_patience =  params['early_stopping_patience'] # params.get('early_stopping_patience', 10)  
    ema_alpha =  params['ema_alpha']
    stabilization_patience =  params['stabilization_patience'] # params.get('stabilization_patience', 5)  # New stabilization check threshold
    
    # # Calculate the number of steps (batches) per epoch
    # steps_per_epoch = sample_size // batch_size
    # # Calculate evaluations per epoch based on eval_freq
    # evaluations_per_epoch = steps_per_epoch // eval_freq
    # # Set stabilization_patience as a multiple of evaluations per epoch
    # stabilization_patience = evaluations_per_epoch //  params['stabilization_patience']
    
    reinitializations_allowed =  params['reinitializations_allowed'] # params.get('reinitializations_allowed', 2)  # Limit reinitializations

    # Initialize models
    nn_stage1 = initialize_and_prepare_model(1, params)
    nn_stage2 = initialize_and_prepare_model(2, params)

    # Check for the initializer type in params and apply accordingly
    if params['initializer'] == 'he':
        nn_stage1.he_initializer(seed=seed_value)  # He initialization (aka Kaiming initialization)
        nn_stage2.he_initializer(seed=seed_value)  # He initialization (aka Kaiming initialization)

    else:
        nn_stage1.reset_weights()  # Custom reset weights check the NN class
        nn_stage2.reset_weights()  

    

    # Initialize optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

    # Prepare training and validation data
    train_data = {
        'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2],
        'A1': tuple_train[5], 'A2': tuple_train[6]
    }
    val_data = {
        'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2],
        'A1': tuple_val[5], 'A2': tuple_val[6]
    }

    # Create a generator for DataLoader
    data_gen = torch.Generator()
    data_gen.manual_seed(seed_value)

    # Create datasets and data loaders
    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=data_gen  # Control shuffling with fixed seed
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=data_gen  # Control shuffling with fixed seed
    )


    # Training variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_stage1_params = None
    best_model_stage2_params = None
    epoch_num_model = 0
    total_steps = 0
    no_improvement_count = 0  # Early stopping counter
    ema_val_loss = None
    reinitialization_count = 0  # Track reinitializations
    
    early_stop = False  # Set the flag to break the outer loop



    # For tabular display
    loss_table = pd.DataFrame(columns=['Epoch', 'Avg Training Loss', 'Avg Validation Loss'])

    for epoch in range(n_epoch):
        
        if early_stop:
            break  # Break out of the epoch loop if early stopping is triggered

        nn_stage1.train()
        nn_stage2.train()

        # Accumulate loss over batches for the epoch
        running_train_loss = 0.0
        running_val_loss = 0.0
        num_batches = 0
        num_val_steps = 0

        # train_loader = get_data_loader_with_reshuffle(train_dataset, batch_size, batches_to_sample)
        for batch_data in train_loader:   

            total_steps += 1
            num_batches += 1

            batch_data = {k: v.to(device) for k, v in batch_data.items()}

            # Forward pass
            outputs_stage1 = nn_stage1(batch_data['input1'])
            outputs_stage2 = nn_stage2(batch_data['input2'])

            outputs_stage1 = torch.stack(outputs_stage1, dim=1).squeeze()
            outputs_stage2 = torch.stack(outputs_stage2, dim=1).squeeze()

            # Compute loss
            loss = main_loss_gamma(
                outputs_stage1, outputs_stage2, batch_data['A1'], batch_data['A2'], 
                batch_data['Ci'], option=option_sur, surrogate_num=params['surrogate_num']
            )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (to prevent exploding gradients)
            if params['gradient_clipping']:
                torch.nn.utils.clip_grad_norm_(nn_stage1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(nn_stage2.parameters(), max_norm=1.0)

            optimizer.step() 
            running_train_loss += loss.item()

            # Evaluate model at specified frequency
            if total_steps % eval_freq == 0:            
                val_loss = evaluate_model(nn_stage1, nn_stage2, val_loader, params)
                running_val_loss += val_loss
                num_val_steps += 1                     

                # Save the best model
                # if val_loss < best_val_loss:
                #     epoch_num_model = epoch
                #     best_val_loss = val_loss
                #     best_model_stage1_params = copy.deepcopy(nn_stage1.state_dict())
                #     best_model_stage2_params = copy.deepcopy(nn_stage2.state_dict())
                #     no_improvement_count = 0  # Reset early stopping counter
                
                # Calculate EMA of validation loss
                if ema_val_loss is None:
                    ema_val_loss = val_loss
                else:
                    ema_val_loss = ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss
                
                # Save the best model using EMA of validation loss
                if ema_val_loss < best_val_loss:
                    # print(" Improved ---> ema_val_loss, best_val_loss: Saving the model...",  ema_val_loss, best_val_loss) 
                    print(f"Improved ---> ema_val_loss: {ema_val_loss}, best_val_loss: {best_val_loss}. Saving the model...")

                    epoch_num_model = epoch
                    best_val_loss = ema_val_loss
                    best_model_stage1_params = copy.deepcopy(nn_stage1.state_dict())
                    best_model_stage2_params = copy.deepcopy(nn_stage2.state_dict())
                    no_improvement_count = 0  # Reset early stopping counter                    
                else:
                    # print(" Did not improve ---> ema_val_loss, best_val_loss, no_improvement_count, stabilization_patience",  ema_val_loss, best_val_loss, no_improvement_count, stabilization_patience)
                    print(f"Did not improve ---> ema_val_loss: {ema_val_loss}, best_val_loss: {best_val_loss}, no_improvement_count: {no_improvement_count}, stabilization_patience: {stabilization_patience}")

                    no_improvement_count += 1  # Increment early stopping counter
                    
                    
                # Check stabilization condition
                if reinitialization_count < reinitializations_allowed and no_improvement_count >= stabilization_patience:
                    print(f"Validation loss stabilized <<<<<<<<<<---------->>>>>>>>>> reinitializing model at epoch {epoch + 1}")
                    
                    nn_stage1 = initialize_and_prepare_model(1, params)  # Reinitialize model 1
                    nn_stage2 = initialize_and_prepare_model(2, params)  # Reinitialize model 2
                    optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)
                    no_improvement_count = 0  # Reset counter after reinitialization
                    reinitialization_count += 1
                    
                 # Early stopping condition based on best_val_loss
                if params['early_stopping'] and reinitialization_count >= reinitializations_allowed and no_improvement_count >= stabilization_patience:
                    print(f"Early stopping after {epoch + 1} epochs due to no further improvement.")                 
                    early_stop = True  # Set the flag to break the outer loop
                    break

                # # Early stopping check 
                # if params['early_stopping'] and no_improvement_count >= early_stopping_patience:
                #     print(f"Early stopping triggered after {epoch + 1} epochs.")
                #     break

                # Update scheduler if necessary
                update_scheduler(scheduler, params, val_loss)                 
                # update_scheduler(scheduler, params, ema_val_loss)


        # Calculate and store average training loss for this epoch
        avg_train_loss = running_train_loss / num_batches
        train_losses.append(avg_train_loss)  
        print(" total_steps   ---->  ",  total_steps)  
        print(" num_batches   ---->  ",  num_batches)                 
        print(" num_val_steps   ---->  ",  num_val_steps)


        if num_val_steps > 0:
            avg_val_loss = running_val_loss / num_val_steps
            val_losses.append(avg_val_loss)
        else:
            val_losses.append(float('nan'))  # In case there were no validation steps this epoch
            
        print()
        print(f'Epoch [{epoch+1}/{n_epoch}], Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss if num_val_steps > 0 else "N/A"}')
        print("_"*90)
        
        # Append to the table
        new_row = pd.DataFrame({
            'Epoch': [epoch + 1],
            'Avg Training Loss': [avg_train_loss],
            'Avg Validation Loss': [avg_val_loss if num_val_steps > 0 else "N/A"]
        })
        loss_table = pd.concat([loss_table, new_row], ignore_index=True)

    

    # Print the table at the end of training
    print()
    print("LOSS TABLE: ")
    print(loss_table.to_string(index=False))
    print()

    # Save the best model parameters
    model_dir = f"models/{params['job_id']}"
    os.makedirs(model_dir, exist_ok=True)

    # Define file paths for saving models
    model_path_stage1 = os.path.join(
        model_dir, f'best_model_stage_surr_1_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')
    
    model_path_stage2 = os.path.join(
        model_dir, f'best_model_stage_surr_2_{sample_size}_config_number_{config_number}_ensemble_num_{ensemble_num}.pt')


    # Ensure the model parameters are not None before saving
    if best_model_stage1_params is not None and best_model_stage2_params is not None:
        # Save models
        torch.save(best_model_stage1_params, model_path_stage1)
        torch.save(best_model_stage2_params, model_path_stage2)

        # Verify that files were saved correctly
        if os.path.isfile(model_path_stage1) and os.path.getsize(model_path_stage1) > 0:
            print(f"Model stage 1 saved successfully at {model_path_stage1}")
        else:
            print(f"Error: Model stage 1 was not saved correctly at {model_path_stage1}")

        if os.path.isfile(model_path_stage2) and os.path.getsize(model_path_stage2) > 0:
            print(f"Model stage 2 saved successfully at {model_path_stage2}")
        else:
            print(f"Error: Model stage 2 was not saved correctly at {model_path_stage2}")
    else:
        print("Error: Model parameters are None, not saving the models.")

    return ((train_losses, val_losses), epoch_num_model)



def DQlearning(tuple_train, tuple_val, params, config_number, seed_value):
    train_input_stage1, train_input_stage2, _, train_Y1, train_Y2, train_A1, train_A2 = tuple_train
    val_input_stage1, val_input_stage2, _, val_Y1, val_Y2, val_A1, val_A2 = tuple_val

    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(params, 1)
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(params, 2)

    # Check for the initializer type in params and apply accordingly
    if params['initializer'] == 'he':
        nn_stage1.he_initializer(seed=seed_value)  # He initialization (aka Kaiming initialization)
        nn_stage2.he_initializer(seed=seed_value)  # He initialization (aka Kaiming initialization)

    else:
        nn_stage1.reset_weights()  # Custom reset weights check the NN class
        nn_stage2.reset_weights()  


    train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate(config_number, nn_stage2, optimizer_2, scheduler_2, 
                                                                                   train_input_stage2, train_A2, train_Y2, 
                                                                                   val_input_stage2, val_A2, val_Y2, params, seed_value+2345, 2)

    train_Y1_hat = evaluate_model_on_actions(nn_stage2, train_input_stage2, train_A2) + train_Y1
    val_Y1_hat = evaluate_model_on_actions(nn_stage2, val_input_stage2, val_A2) + val_Y1

    train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate(config_number, nn_stage1, optimizer_1, scheduler_1, 
                                                                                   train_input_stage1, train_A1, train_Y1_hat, 
                                                                                   val_input_stage1, val_A1, val_Y1_hat, params, seed_value+123, 1)

    return (train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2)



def evaluate_tao(S1, S2, d1_star, d2_star, params_ds, config_number):

    # Convert test input from PyTorch tensor to numpy array
    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    # Load the R script that contains the required function
    ro.r('source("ACWL_tao.R")')

    # Convert S2 to the same data type as S1
    S2 =   S2.astype(S1.dtype) #  S2.reshape(-1, 1) #


    # Convert tensors to NumPy arrays
    gamma1_np = params_ds['gamma1'].cpu().numpy().reshape(-1, 1)  # Reshape to be a column vector
    gamma1_prime_np = params_ds['gamma1_prime'].cpu().numpy().reshape(-1, 1)
    delta_A1_np = params_ds['delta_A1'].cpu().numpy()
    eta_A1_np = params_ds['eta_A1'].cpu().numpy()

    gamma2_np = params_ds['gamma2'].cpu().numpy().reshape(-1, 1)  # Reshape to be a column vector
    gamma2_prime_np = params_ds['gamma2_prime'].cpu().numpy().reshape(-1, 1)

    delta_A2_np = params_ds['delta_A2'].cpu().numpy()
    eta_A2_np = params_ds['eta_A2'].cpu().numpy()

    lambda_val_np = params_ds['lambda_val'].item()



    # Call the R function with the parameters
    results = ro.globalenv['test_ACWL'](S1, S2, d1_star.cpu().numpy(), d2_star.cpu().numpy(), params_ds['noiseless'], 
                                        config_number, params_ds['job_id'], param_m1 = params_ds['m1'], param_m2 = params_ds['m2'],
                                        setting=params_ds['setting'],
                                        func = params_ds["f"], neu = params_ds["neu"], alpha = params_ds["alpha"], u = params_ds["u"],
                                        gamma1=gamma1_np, 
                                        gamma1_prime=gamma1_prime_np, 
                                        delta_A1=delta_A1_np, 
                                        eta_A1=eta_A1_np, 
                                        gamma2=gamma2_np, 
                                        gamma2_prime=gamma2_prime_np, 
                                        delta_A2=delta_A2_np, 
                                        eta_A2=eta_A2_np,         
                                        # lambda_val=lambda_val_np
                                        )

    # Extract the decisions and convert to PyTorch tensors on the specified device
    A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.int64).to(params_ds['device'])
    A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.int64).to(params_ds['device'])

    return A1_Tao, A2_Tao



# def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):
def eval_DTR(V_replications, num_replications, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):

    # Generate and preprocess data for evaluation
    processed_result = generate_and_preprocess_data(params_ds, replication_seed=num_replications+1234, config_seed=config_number, run='test')


    test_input_stage1, test_input_stage2, test_O2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2, Y1_g1_opt, Y2_g2_opt  = processed_result



    # # Debug print for processed_result
    # print("========== DEBUG: Eval processed_result ==========")
    # print(test_input_stage1)
    # print("=============================================")


    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]

    # Append policy values for DS
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu())  
    V_replications["V_replications_M1_Optimal"].append(torch.mean(Y1_g1_opt + Y2_g2_opt).cpu())  

    # # Value function behavioral
    # message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} '
    # print(message)

    #######################################
    # Evaluation phase using Tao's method #
    #######################################
    if params_ds.get('run_adaptive_contrast_tao', True):
        start_time = time.time()  # Start time recording
        # A1_Tao, A2_Tao = evaluate_tao(test_input_stage1, test_O2, A1_tensor_test, A2_tensor_test, Y1_tensor, Y2_tensor, params_ds, config_number)
        A1_Tao, A2_Tao = evaluate_tao(test_input_stage1, test_O2, d1_star, d2_star, params_ds, config_number)

        end_time = time.time()  # End time recording
        print(f"Total time taken to run evaluate_tao: { end_time - start_time} seconds")
        

        # Append to DataFrame
        new_row_Tao = {
            'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
            'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
            'Predicted_A1': A1_Tao.cpu().numpy().tolist(),
            'Predicted_A2': A2_Tao.cpu().numpy().tolist()
        }
        df_Tao = pd.concat([df_Tao, pd.DataFrame([new_row_Tao])], ignore_index=True)

        # Calculate policy values fn. using the estimator of Tao's method
        # print("Tao's method estimator: ")
        start_time = time.time()  # Start time recording
        V_rep_Tao = calculate_policy_valuefunc("Tao", test_input_stage1, test_O2, params_ds, A1_Tao, A2_Tao, d1_star, d2_star, Z1, Z2)
        # V_rep_Tao = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_Tao, A2_Tao, P_A1_g_H1, P_A2_g_H2, config_number)
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run estimator function for testing: { end_time - start_time} seconds")
                
        # Append policy values for Tao
        V_replications["V_replications_M1_pred"]["Tao"].append(V_rep_Tao)     
        message = f'\nY1_tao+Y2_tao mean: {V_rep_Tao} \n'
        print(message)

    #######################################
    # Evaluation phase using DQL's method #
    #######################################
    if params_ds.get('run_DQlearning', True):
        start_time = time.time()  # Start time recording
        df_DQL, V_rep_DQL = evaluate_method_DQL('DQL', params_dql, config_number, df_DQL, test_input_stage1, A1_tensor_test, test_O2, test_input_stage2, 
                                            A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2 )
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run evaluate_method ('DQL'): { end_time - start_time} seconds")
        # Append policy values for DQL
        V_replications["V_replications_M1_pred"]["DQL"].append(V_rep_DQL)     
        message = f'\nY1_DQL+Y2_DQL mean: {V_rep_DQL} '
        print(message)

    ########################################
    #  Evaluation phase using DS's method  #
    ########################################
    if params_ds.get('run_surr_opt', True):
        start_time = time.time()  # Start time recording
        df_DS, V_rep_DS = evaluate_method_DS('DS', params_ds, config_number, df_DS, test_input_stage1, A1_tensor_test, test_O2, test_input_stage2, 
                                        A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2 )
        end_time = time.time()  # End time recording
        print(f"\nTotal time taken to run evaluate_method ('DS'): { end_time - start_time} seconds\n")
                    
        # Append policy values for DS
        V_replications["V_replications_M1_pred"]["DS"].append(V_rep_DS)
        message = f'Y1_DS+Y2_DS mean: {V_rep_DS}'
        print(message)

    return V_replications, df_DQL, df_DS, df_Tao 



def adaptive_contrast_tao(all_data, contrast, config_number, job_id, setting):
    S1, S2, train_Y1, train_Y2, train_A1, train_A2, pi_tensor_stack, g1_opt, g2_opt = all_data

    # Convert all tensors to CPU and then to NumPy
    A1 = train_A1.cpu().numpy()
    probs1 = pi_tensor_stack.T[:, :3].cpu().numpy()

    A2 = train_A2.cpu().numpy()
    probs2 = pi_tensor_stack.T[:, 3:].cpu().numpy()

    R1 = train_Y1.cpu().numpy()
    R2 = train_Y2.cpu().numpy()

    g1_opt = g1_opt.cpu().numpy()
    g2_opt = g2_opt.cpu().numpy()

    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    # Load the R script containing the function
    ro.r('source("ACWL_tao.R")')

    # Call the R function with the numpy arrays
    results = ro.globalenv['train_ACWL'](job_id, S1, S2, A1, A2, probs1, probs2, R1, R2, g1_opt, g2_opt, config_number, contrast, setting)

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

    # Generate and preprocess data for training
    # params['gamma1'] = torch.randn(params['input_dim'], device=device)
    # params['gamma2'] = torch.randn(params['input_dim'], device=device)
    # params['gamma1_prime'] = torch.randn(params['input_dim'], device=device)
    # params['gamma2_prime'] = torch.randn(params['input_dim'], device=device)

    # All ones for gamma vectors
    params['gamma1'] = torch.ones(params['input_dim'], device=device)
    params['gamma2'] = torch.ones(params['input_dim'], device=device)
    params['gamma1_prime'] = torch.ones(params['input_dim'], device=device)
    params['gamma2_prime'] = torch.ones(params['input_dim'], device=device)

    # # Scaled identity-based vectors (linear scaling)
    # params['gamma1'] = torch.tensor([i / (params['input_dim']**0.5) for i in range(1, params['input_dim'] + 1)], device=device)
    # params['gamma2'] = torch.tensor([i / (params['input_dim']**0.5) for i in range(1, params['input_dim'] + 1)], device=device)
    # params['gamma1_prime'] = torch.tensor([i / (params['input_dim']**0.5) for i in range(params['input_dim'], 0, -1)], device=device)
    # params['gamma2_prime'] = torch.tensor([i / (params['input_dim']**0.5) for i in range(params['input_dim'], 0, -1)], device=device)
    
    # params['delta_A1'] =  torch.tensor([1.0, 10.0, 1.0], device=device)  # Amplify difference, Optimal action is A1=2
    # params['delta_A2'] =  torch.tensor([1.0, 1.0, 10.0], device=device)  # Optimal action is A2=2
    

    # params['eta_A1'] = torch.tensor([1.0, 10.0, 1.0], device=device) 
    # params['eta_A2'] = torch.tensor([1.0, 1.0, 10.0], device=device) 
    
    if params["t_difficulty"] == "easy":
        # Easy: Values are far apart, minimal dependency
        params['delta_A1'] = torch.tensor([5.0, 1.0, 3.0], device=device)
        params['delta_A2'] = torch.tensor([4.5, 2.0, 5.0], device=device)

        # params['lambda_val'] = torch.tensor(0.1, device=device)  # Minimal dependency
        params['lambda_val'] = torch.tensor(0.3, device=device)  # Moderate dependency

        params['eta_A1'] = torch.tensor([3.0, 1.0, 4.0], device=device)
        params['eta_A2'] = torch.tensor([5.0, 1.5, 3.5], device=device)

    elif params["t_difficulty"] == "medium":
#         # Medium: Values are moderately spaced, moderate dependency
#         params['delta_A1'] = torch.tensor([3.0, 2.0, 2.5], device=device)
#         params['delta_A2'] = torch.tensor([2.8, 2.5, 3.2], device=device)

#         params['lambda_val'] = torch.tensor(0.3, device=device)  # Moderate dependency

#         params['eta_A1'] = torch.tensor([2.5, 2.0, 3.0], device=device)
#         params['eta_A2'] = torch.tensor([3.0, 2.0, 3.5], device=device)
        

        params['delta_A1'] =  torch.tensor([2.0, 3.0, 1.5], device=device) # Amplify difference, Optimal action is A1= 2
        params['delta_A2'] =  torch.tensor([2.5, 1.5, 3.0], device=device) # Optimal action is A2= 3

        params['lambda_val'] = torch.tensor(0.3, device=device) # Dependency on Y1 in Y2

        params['eta_A1'] = torch.tensor([1.0, 2.5, 2.0], device=device)
        params['eta_A2'] = torch.tensor([2.0, 1.0, 2.5], device=device)


    else:  # Difficult: Values are close together, high dependency
        params['delta_A1'] = torch.tensor([2.5, 2.7, 2.6], device=device)
        params['delta_A2'] = torch.tensor([2.6, 2.5, 2.7], device=device)

        # params['lambda_val'] = torch.tensor(0.5, device=device)  # Strong dependency
        params['lambda_val'] = torch.tensor(0.3, device=device)  # Moderate dependency

        params['eta_A1'] = torch.tensor([2.1, 2.0, 2.2], device=device)
        params['eta_A2'] = torch.tensor([2.2, 2.1, 2.3], device=device)

    
    




    losses_dict = {'DQL': {}, 'DS': {}} 
    epoch_num_model_lst = []

    print("config_grid: ", params, "\n\n")

    # Clone the updated config for DQlearning and surr_opt
    params_DQL_u = copy.deepcopy(params)
    params_DS_u = copy.deepcopy(params)
    
    params_DS_u['f_model'] = 'surr_opt'
    params_DQL_u['f_model'] = 'DQlearning'
    params_DQL_u['num_networks'] = 1  

    for replication in tqdm(range(params['num_replications']), desc="Replications_M1"):
        print(f"\nReplication # -------------->>>>>  {replication+1}")

        seed_value = config_number * 100 + replication 

        # Generate and preprocess data for training
        tuple_train, tuple_val, adapC_tao_Data = generate_and_preprocess_data(params, replication_seed=replication, config_seed=config_number, run='train')

        # # Debug print for processed_result
        # print("========== DEBUG: Train tuple_train ==========")
        # print(tuple_train[0])
        # print("=============================================")


        # # Debug print for processed_result
        # print("========== DEBUG: Train tuple_val ==========")
        # print(tuple_val[0])
        # print("=============================================")



        # Estimate treatment regime : model --> surr_opt
        print("Training started!")
        
        # Run ALL models on the same tuple of data
        if params.get('run_adaptive_contrast_tao', True):
            start_time = time.time()  # Start time recording
            # (select2, select1, selects) = adaptive_contrast_tao(adapC_tao_Data, params["contrast"], config_number, params["job_id"])
            (select2, select1, selects) = adaptive_contrast_tao(adapC_tao_Data, params["contrast"], config_number, params["job_id"], params["setting"])
            end_time = time.time()  # End time recording
            print(f"Total time taken to run adaptive_contrast_tao: { end_time - start_time} seconds")
            
        if params.get('run_DQlearning', True):
            # Run both models on the same tuple of data
            params_DQL_u['input_dim_stage1'] = params['input_dim_stage1'] + 1 # Ex. TAO: 5 + 1 = 6 # (H_1, A_1)
            params_DQL_u['input_dim_stage2'] = params['input_dim_stage2'] + 1 # Ex. TAO: 7 + 1 = 8 # (H_2, A_2)

            start_time = time.time()  # Start time recording
            trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL_u, config_number, seed_value)
            end_time = time.time()  # End time recording
            print(f"Total time taken to run DQlearning: { end_time - start_time} seconds")
            # Store losses 
            losses_dict['DQL'][replication] = trn_val_loss_tpl_DQL
            
        if params.get('run_surr_opt', True):
            params_DS_u['input_dim_stage1'] = params['input_dim_stage1']  # Ex. TAO: 5  # (H_1, A_1)
            params_DS_u['input_dim_stage2'] = params['input_dim_stage2']  # Ex. TAO: 7  # (H_2, A_2)

            start_time = time.time()  # Start time recording

            for ensemble_num in range(params['ensemble_count']):
                print()
                print(f"***************************************** Train -> Agent #: {ensemble_num}*****************************************")
                print()
                if params['phi_ensemble']:
                    option_sur = params['option_sur']
                else:
                    option_sur = ensemble_num+1
                trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS_u, config_number, ensemble_num, option_sur, seed_value)

            end_time = time.time()  # End time recording 
            print(f"Total time taken to run surr_opt: { end_time - start_time} seconds")
            # Append epoch model results from surr_opt
            epoch_num_model_lst.append(epoch_num_model_DS)
            # Store losses 
            losses_dict['DS'][replication] = trn_val_loss_tpl_DS
            
        # eval_DTR 
        print("Evaluation started") 
        start_time = time.time()  # Start time recording 
        V_replications, df_DQL, df_DS, df_Tao = eval_DTR(V_replications, replication, df_DQL, df_DS, df_Tao, params_DQL_u, params_DS_u, config_number)
        end_time = time.time()  # End time recording
        print(f"Total time taken to run eval_DTR: { end_time - start_time} seconds \n\n")
                
    return V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst


def run_training(config, config_updates, V_replications, config_number, config_replication_seed):
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = simulations(V_replications, local_config, config_number)
    
    if not any(V_replications[key] for key in V_replications):
        warnings.warn("V_replications is empty. Skipping accuracy calculation.")
    else:
        VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh, VF_df_Opt = extract_value_functions_separate(V_replications)
        return VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh, VF_df_Opt, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst
 

# parallelized 

def run_training_with_params(params):

    config, current_config, V_replications, i, config_number = params
    return run_training(config, current_config, V_replications, config_number, config_replication_seed=i)
 


        
def run_grid_search(config, param_grid):
    # Initialize for storing results and performance metrics
    results = {}
    # Initialize separate cumulative DataFrames for DQL and DS
    all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
    all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run
    all_dfs_Tao = pd.DataFrame()   # DataFrames from each Tao run

    all_losses_dicts = []  # Losses from each run
    all_epoch_num_lists = []  # Epoch numbers from each run 

    # Initialize empty lists to store the value functions across all configurations
    all_performances_DQL = []
    all_performances_DS = []
    all_performances_Tao = []
    all_performances_Beh = []
    all_performances_Opt = []

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
                    "V_replications_M1_Optimal": []
                }
                params = (config, current_config, V_replications, i, config_number)
                future = executor.submit(run_training_with_params, params)
                future_to_params[future] = (current_config, i)

        for future in concurrent.futures.as_completed(future_to_params):
            current_config, i = future_to_params[future]            
            performance_DQL, performance_DS, performance_Tao, performance_Beh, performance_Opt, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = future.result()
            
            print(f'Configuration {current_config}, replication {i} completed successfully.')
            
            # Processing performance DataFrame for both methods
            performances_DQL = pd.DataFrame()
            performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

            performances_DS = pd.DataFrame()
            performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            performances_Tao = pd.DataFrame()
            performances_Tao = pd.concat([performances_Tao, performance_Tao], axis=0)

            performances_Beh = pd.DataFrame()
            performances_Beh = pd.concat([performances_Beh, performance_Beh], axis=0)

            performances_Opt = pd.DataFrame()
            performances_Opt = pd.concat([performances_Opt, performance_Opt], axis=0)

            # Process and store DQL performance
            dql_values = [value.item() if value is not None else None for value in performances_DQL['Method\'s Value fn.']]
            all_performances_DQL.append(dql_values)

            # Process and store DS performance
            ds_values = [value.item() if value is not None else None for value in performances_DS['Method\'s Value fn.']]
            all_performances_DS.append(ds_values)

            # Process and store Tao performance
            tao_values = [value.item() if value is not None else None for value in performances_Tao['Method\'s Value fn.']]
            all_performances_Tao.append(tao_values)

            # Process and store Behavioral performance
            beh_values = [value.item() if value is not None else None for value in performances_Beh['Method\'s Value fn.']]
            all_performances_Beh.append(beh_values)

            # Process and store Tao performance
            opt_values = [value.item() if value is not None else None for value in performances_Opt['Method\'s Value fn.']]
            all_performances_Opt.append(opt_values)



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
            performance_Beh_mean = performances_Beh["Method's Value fn."].mean()
            performance_Opt_mean = performances_Opt["Method's Value fn."].mean()


            # Calculating the standard deviation for "Method's Value fn."
            performance_DQL_std = performances_DQL["Method's Value fn."].std()
            performance_DS_std = performances_DS["Method's Value fn."].std()
            performance_Tao_std = performances_Tao["Method's Value fn."].std()
            performance_Beh_std = performances_Beh["Method's Value fn."].std()
            performance_Opt_std = performances_Opt["Method's Value fn."].std()

            # Check if the configuration key exists in the results dictionary
            if config_key not in results:
                # If not, initialize it with dictionaries for each model containing the mean values
                results[config_key] = {
                    'Behavioral': {"Method's Value fn.": performance_Beh_mean, 
                           "Method's Value fn. SD": performance_Beh_std,
                           },
                    'Optimal': {"Method's Value fn.": performance_Opt_mean, 
                           "Method's Value fn. SD": performance_Opt_std,
                           },
                    'DQL': {"Method's Value fn.": performance_DQL_mean, 
                            "Method's Value fn. SD": performance_DQL_std, 
                            },
                    'DS': {"Method's Value fn.": performance_DS_mean, 
                           "Method's Value fn. SD": performance_DS_std,
                           },
                    'Tao': {"Method's Value fn.": performance_Tao_mean, 
                           "Method's Value fn. SD": performance_Tao_std,
                           }   
                }
            else:
                # Update existing entries with new means
                results[config_key]['DQL'].update({
                    "Method's Value fn.": performance_DQL_mean,                                 
                    "Method's Value fn. SD": performance_DQL_std, 
                })
                results[config_key]['DS'].update({
                    "Method's Value fn.": performance_DS_mean,
                    "Method's Value fn. SD": performance_DS_std,
                })
                results[config_key]['Tao'].update({
                    "Method's Value fn.": performance_Tao_mean, 
                    "Method's Value fn. SD": performance_Tao_std,
                })
                results[config_key]['Behavioral'].update({
                    "Method's Value fn.": performance_Beh_mean, 
                    "Method's Value fn. SD": performance_Beh_std,  
                })
                results[config_key]['Optimal'].update({
                    "Method's Value fn.": performance_Opt_mean, 
                    "Method's Value fn. SD": performance_Opt_std,  
                })                  

            print("Performances for configuration: %s", config_key)
            print("performance_DQL_mean: %s", performance_DQL_mean)
            print("performance_DS_mean: %s", performance_DS_mean)
            print("performance_Tao_mean: %s", performance_Tao_mean)
            print("\n\n")
                
    folder = f"data/{config['job_id']}"
    save_simulation_data(all_performances_Beh, all_performances_Opt, all_performances_DQL, all_performances_DS,  all_performances_Tao, all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder)
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
    
    
    
    
    
        
def main(job_id):

    # Load configuration and set up the device
    # config = load_config('config_newScheme.yaml') 
    # config = load_config('config_tao.yaml')  
    config = load_config('config.yaml') 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    
    
    # # Get the SLURM_JOB_ID from environment variables
    # job_id = os.getenv('SLURM_JOB_ID')

    # # If job_id is None, set it to the current date and time formatted as a string
    # if job_id is None:
    #     job_id = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YYYYMMDDHHMMSS
    
    config['job_id'] = job_id
    
    print("Job ID: ", job_id)

    # training_validation_prop = config['training_validation_prop']
    # train_size = int(training_validation_prop * config['sample_size'])
    # print("Training size: %d", train_size)   

    
    
    # # Define parameter grid for grid search
    # Empty Grid
    param_grid = {}
    
    # param_grid = {  
    #     'option_sur': [1,2,3,4,5], 
    #     }

    # param_grid = {  
    #     'dropout_rate': [0.1,0.3,0.4], 
    #     'factor': [0.8,0.5,0.1], 
    #     'ema_alpha': [ 0.1, 0.4, 0.8],
    #     'optimizer_lr':[0.01, 0.07, 0.1]
    #     }

    # param_grid = {  
    #     # 'dropout_rate': [0.1, 0.3, 0.4], #0.3, 0.43
    #     'factor': [0.8, 0.7, 0.5, 0.2], 
    #     'ema_alpha': [0.05, 0.1, 0.3]
    #     # 'optimizer_lr':[0.01, 0.07, 0.1]
    #     }

    # param_grid = {  
    #     'batch_size': [100, 200, 400, 700, 1200]
    #     }

    # param_grid = {
    #     'activation_function': [ 'elu', 'relu'], # 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'
    #     'learning_rate': [0.07], # 0.07
    #     'num_layers': [4], # 2, 4, => 0 means truly linear model, here num_layers means --> number of hidden layers
    #     'batch_size': [300, 800], #300
    #     'hidden_dim_stage1': [40],  #5,10  Number of neurons in the hidden layer of stage 1
    #     'hidden_dim_stage2': [40],  #5,10  Number of neurons in the hidden layer of stage 2 
    #     'dropout_rate': [0.4],  # 0, 0.1, 0.4 Dropout rate to prevent overfitting
    #     'n_epoch': [60, 150], #60, 150
    # }


    # param_grid = {
    #     'activation_function': [ 'elu'], # 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'
    #     'learning_rate': [0.07], # 0.07
    #     'num_layers': [4], # 2, 4, => 0 means truly linear model, here num_layers means --> number of hidden layers
    #     'batch_size': [800], #300
    #     'hidden_dim_stage1': [40],  #5,10  Number of neurons in the hidden layer of stage 1
    #     'hidden_dim_stage2': [40],  #5,10  Number of neurons in the hidden layer of stage 2 
    #     'dropout_rate': [0.4],  # 0, 0.1, 0.4 Dropout rate to prevent overfitting
    #     'n_epoch': [60], #60, 150
    # }
    
    # param_grid = {
    #     'activation_function': [ 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'], # 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'
    #     'learning_rate': [0.01], # 0.07
    #     'num_layers': [4], # 2, 4, => 0 means truly linear model, here num_layers means --> number of hidden layers
    #     'batch_size': [200, 800], #300
    #     'add_ll_batch_norm': [True],  #5,10  Number of neurons in the hidden layer of stage 1
    #     'dropout_rate': [0.4],  # 0, 0.1, 0.4 Dropout rate to prevent overfitting
    #     'n_epoch': [10], #20, 60, 150
    # }
    
    
    
    if config['run_adaptive_contrast_tao']:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        # Load the R script to avoid dynamic loading
        ro.r.source("ACWL_tao.R")


    # Perform operations whose output should go to the file
    run_grid_search(config, param_grid)
    




# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     print(f'Total time taken: {end_time - start_time:.2f} seconds')

     




class FlushFile:
    """File-like wrapper that flushes on every write and writes to both console and a file."""
    def __init__(self, f, logfile):
        self.f = f
        self.logfile = logfile

    def write(self, x):
        self.f.write(x)          # Write to terminal
        self.f.flush()           # Flush terminal output
        self.logfile.write(x)    # Write to the log file
        self.logfile.flush()     # Flush file output

    def flush(self):
        self.f.flush()
        self.logfile.flush()






if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # Get the SLURM_JOB_ID from environment variables
    job_id = os.getenv('SLURM_JOB_ID')

    # If job_id is None, set it to the current date and time formatted as a string
    if job_id is None:
        job_id = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YYYYMMDDHHMMSS

    # Ensure the 'data/job_id' directory exists
    log_dir = os.path.join('data', job_id)
    os.makedirs(log_dir, exist_ok=True)  # Creates the directory if it doesn't exist

    # Create a timestamp-based filename
    log_time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))  # e.g., '20231026_142530'
    log_filename = f"output_log_{log_time_str}.txt"
    
    # Full path for the log file
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Open the log file in write mode
    logfile = open(log_filepath, 'w')  # Open outside the 'with' block
    
    # Set stdout to write to both terminal and logfile
    sys.stdout = FlushFile(sys.stdout, logfile)
    
    try:
        # Record the start time
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        print(f'Start time: {start_time_str}')
        
        # Call the main function and pass the job_id
        main(job_id)
        
        # Record the end time
        end_time = time.time()
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        print(f'End time: {end_time_str}')
        
        # Calculate and log the total time taken
        total_time = end_time - start_time
        print(f'Total time taken: {total_time:.2f} seconds')
    
    finally:
        # Reset stdout back to its original state and close the logfile
        sys.stdout = sys.__stdout__  # Reset stdout to its original state
        logfile.close()  # Now it's safe to close the logfile