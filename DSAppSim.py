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


import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()
# Load the R script to avoid dynamic loading
ro.r.source("ACWL_tao.R")





# Generate Data
def generate_and_preprocess_data(params, replication_seed, run='train'):

    # torch.manual_seed(replication_seed)
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
        
        # # Approach 2 S2
        # result2 = A_sim(matrix_pi2, stage=2)
        # if  params['use_m_propen']:
        #     A2, _ = result2['A'], result2['probs']
        #     probs2 = M_propen(A2, input_stage2, stage=2)  # multinomial logistic regression with H2
        # else:         
        #     A2, probs2 = result2['A'], result2['probs']
        # A2 += 1

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
        # input_stage1 = O1
        # params['input_dim_stage1'] = input_stage1.shape[1] # (H_1)  for DS

        # x1, x2, x3, x4, x5 = O1[0], O1[1], O1[2], O1[3], O1[4]
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

        # g1_opt = ((x1 > -1).float() * ((x2 > -0.5).float() + (x2 > 0.5).float())) + 1
        # Y1 = torch.exp(1.5 - torch.abs(1.5 * x1 + 2) * (A1 - g1_opt).pow(2)) + Z1

        g1_opt = ((O1[:, 0] > -1).float() * ((O1[:, 1] > -0.5).float() + (O1[:, 1] > 0.5).float())) + 1
        # Y1 = torch.exp(1.5 - torch.abs(1.5 * O1[:, 0] + 2) * (A1 - g1_opt).pow(2)) + Z1  
        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params) 

        # Input preparation
        # input_stage2 = torch.cat([O1, A1.unsqueeze(1).to(device), Y1.unsqueeze(1).to(device)], dim=1)
        # params['input_dim_stage2'] = input_stage2.shape[1] # 5 # 7 + 1 = 8 # (H_2)

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

        # Y1_opt = torch.exp(torch.tensor(1.5, device=device)) + Z1
        # g2_opt = (x3 > -1).float() * ((Y1_opt > 0.5).float() + (Y1_opt > 3).float()) + 1
        # Y2 = torch.exp(1.26 - torch.abs(1.5 * x3 - 2) * (A2 - g2_opt).pow(2)) + Z2     

        Y1_opt =  torch.exp(torch.tensor(1.5, device=Z1.device)) + Z1
        g2_opt = (O1[:, 2] > -1).float() * ((Y1_opt > 0.5).float() + (Y1_opt > 3).float()) + 1
        # Y2 = torch.exp(1.26 - torch.abs(1.5 * O1[:, 2] - 2) * (A2 - g2_opt).pow(2)) + Z2

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
           
        g1_opt = dt_star(O1)
        g2_opt = torch.argmax(O1**2, dim=1) + 1

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

        Y1 = calculate_reward_stage1(O1, A1, g1_opt, Z1, params)

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

        Y2 = calculate_reward_stage2(O1, A1, O2, A2, g2_opt, Z2, params) 
   
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
      
    if O2.ndimension() == 1 and O2.numel>0: # O2 is 1D, treated as a single column
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
    # print("calculate_reward_stage2: ", O1, g1_opt, O2, g2_opt, g2_opt, Z2)
    Y2_g2_opt = calculate_reward_stage2(O1, g1_opt, O2, g2_opt, g2_opt, Z2, params)

    print()
    print("="*60)
    print("Y1_g1_opt mean: ", torch.mean(Y1_g1_opt) )
    print("Y2_g2_opt mean: ", torch.mean(Y2_g2_opt) )         
    print("Y1_g1_opt+Y2_g2_opt mean: ", torch.mean(Y1_g1_opt+Y2_g2_opt) )
    print("="*60)

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
        return input_stage1, input_stage2, O2, Y1, Y2, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, g1_opt, g2_opt, Z1, Z2

    # Splitting data into training and validation sets
    train_size = int(params['training_validation_prop'] * Y1.shape[0])
    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1, Y2, A1, A2]]

    # return tuple(train_tensors), tuple(val_tensors)
    return tuple(train_tensors), tuple(val_tensors), tuple([O1, O2, Y1, Y2, A1, A2, pi_tensor_stack, g1_opt, g2_opt])


def surr_opt(tuple_train, tuple_val, params, config_number):
    
    sample_size = params['sample_size'] 
    
    train_losses, val_losses = [], []
    best_val_loss, best_model_stage1_params, best_model_stage2_params, epoch_num_model = float('inf'), None, None, 0

    nn_stage1 = initialize_and_prepare_model(1, params)
    nn_stage2 = initialize_and_prepare_model(2, params)

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
    
    return ((train_losses, val_losses), epoch_num_model)



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

    return (train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2)



def evaluate_tao(S1, S2, d1_star, d2_star, params_ds, config_number):

    # Convert test input from PyTorch tensor to numpy array
    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    # Load the R script that contains the required function
    ro.r('source("ACWL_tao.R")')

    # print("S1: ", S1.shape, type(S1))
    # print("S2: ", S2.shape, type(S2)) 

    # Convert S2 to the same data type as S1
    # S2 = S2.to(S1.dtype)  # for torch objects
    S2 =   S2.astype(S1.dtype) #  S2.reshape(-1, 1) #


    # Call the R function with the parameters
    results = ro.globalenv['test_ACWL'](S1, S2, d1_star.cpu().numpy(), d2_star.cpu().numpy(), params_ds['noiseless'], 
                                        config_number, params_ds['job_id'], setting=params_ds['setting'])

    # Extract the decisions and convert to PyTorch tensors on the specified device
    # A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.float32).to(params_ds['device'])
    # A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.float32).to(params_ds['device'])

    A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.int64).to(params_ds['device'])
    A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.int64).to(params_ds['device'])

    return A1_Tao, A2_Tao



# def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):
def eval_DTR(V_replications, num_replications, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):


    # Generate and preprocess data for evaluation
    processed_result = generate_and_preprocess_data(params_ds, replication_seed=num_replications, run='test')
    test_input_stage1, test_input_stage2, test_O2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2  = processed_result
    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]

    # Append policy values for DS
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu().item())  
    # Value function behavioral
    message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} '
    print(message)

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
        print(f"\n\nTotal time taken to run calculate_policy_values_W_estimator_tao: { end_time - start_time} seconds")
                
        # Append policy values for Tao
        V_replications["V_replications_M1_pred"]["Tao"].append(V_rep_Tao)     
        message = f'\nY1_tao+Y2_tao mean: {V_rep_Tao} \n'
        print(message)

    #######################################
    # Evaluation phase using DQL's method #
    #######################################
    if params_ds.get('run_DQlearning', True):
        start_time = time.time()  # Start time recording
        df_DQL, V_rep_DQL = evaluate_method('DQL', params_dql, config_number, df_DQL, test_input_stage1, A1_tensor_test, test_O2, test_input_stage2, 
                                            A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2 )
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run evaluate_method)W_estimator('DQL'): { end_time - start_time} seconds")
        # Append policy values for DQL
        V_replications["V_replications_M1_pred"]["DQL"].append(V_rep_DQL)     
        message = f'\nY1_DQL+Y2_DQL mean: {V_rep_DQL} '
        print(message)

    ########################################
    #  Evaluation phase using DS's method  #
    ########################################
    if params_ds.get('run_surr_opt', True):
        start_time = time.time()  # Start time recording
        df_DS, V_rep_DS = evaluate_method('DS', params_ds, config_number, df_DS, test_input_stage1, A1_tensor_test, test_O2, test_input_stage2, 
                                        A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, d1_star, d2_star, Z1, Z2 )
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run evaluate_method)W_estimator('DS'): { end_time - start_time} seconds\n\n")
                    
        # Append policy values for DS
        V_replications["V_replications_M1_pred"]["DS"].append(V_rep_DS)
        message = f'\nY1_DS+Y2_DS mean: {V_rep_DS} '
        print(message)

    return V_replications, df_DQL, df_DS, df_Tao # {"df_DQL": df_DQL, "df_DS":df_DS, "df_Tao": df_Tao}



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
    # results = ro.globalenv['train_ACWL'](train_input_np, job_id, A1, probs1, A2, probs2, R1, R2, g1_opt, g2_opt, config_number, contrast, method="tao")

    # Extract results
    select2 = results.rx2('select2')[0]
    select1 = results.rx2('select1')[0]
    selects = results.rx2('selects')[0]

    print("select2, select1, selects: ", select2, select1, selects)

    return select2, select1, selects



def simulations(V_replications, params, config_fixed, config_number):

    columns = ['Behavioral_A1', 'Behavioral_A2', 'Predicted_A1', 'Predicted_A2']

    # Initialize separate DataFrames for DQL and DS
    df_DQL = pd.DataFrame(columns=columns)
    df_DS = pd.DataFrame(columns=columns)
    df_Tao = pd.DataFrame(columns=columns)

    losses_dict = {'DQL': {}, 'DS': {}} 
    epoch_num_model_lst = []

    print("config_grid: ", params, "\n\n")
    # print("config_fixed: ", config_fixed, "\n")

    # Clone the updated config for DQlearning and surr_opt
    params_DQL_u = copy.deepcopy(params)
    params_DS_u = copy.deepcopy(params)
    
    params_DS_u['f_model'] = 'surr_opt'
    params_DQL_u['f_model'] = 'DQlearning'
    params_DQL_u['num_networks'] = 1  

    # Clone the fixed config for DQlearning and surr_opt
    config_fixed['num_layers'] = params['num_layers']
    config_fixed['hidden_dim_stage1'] = params['hidden_dim_stage1']
    config_fixed['hidden_dim_stage2'] = params['hidden_dim_stage2']
    config_fixed['activation_function'] = params['activation_function']

    params_DQL_f = copy.deepcopy(config_fixed)
    params_DS_f = copy.deepcopy(config_fixed)
    
    params_DS_f['f_model'] = 'surr_opt'
    params_DQL_f['f_model'] = 'DQlearning'
    params_DQL_f['num_networks'] = 1  


    for replication in tqdm(range(params['num_replications']), desc="Replications_M1"):
        print(f"\nReplication # -------------->>>>>  {replication+1}")

        # Generate and preprocess data for training
        tuple_train, tuple_val, adapC_tao_Data = generate_and_preprocess_data(params, replication_seed=replication, run='train')

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

            params_DQL_f['input_dim_stage1'] = params['input_dim_stage1'] + 1 # Ex. TAO: 5 + 1 = 6 # (H_1, A_1)
            params_DQL_f['input_dim_stage2'] = params['input_dim_stage2'] + 1 # Ex. TAO: 7 + 1 = 8 # (H_2, A_2)

            start_time = time.time()  # Start time recording
            trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL_u, config_number)
            end_time = time.time()  # End time recording
            print(f"Total time taken to run DQlearning: { end_time - start_time} seconds")
            # Store losses 
            losses_dict['DQL'][replication] = trn_val_loss_tpl_DQL
            
        if params.get('run_surr_opt', True):
            params_DS_u['input_dim_stage1'] = params['input_dim_stage1']  # Ex. TAO: 5  # (H_1, A_1)
            params_DS_u['input_dim_stage2'] = params['input_dim_stage2']  # Ex. TAO: 7  # (H_2, A_2)

            params_DS_f['input_dim_stage1'] = params['input_dim_stage1']  # Ex. TAO: 5  # (H_1, A_1)
            params_DS_f['input_dim_stage2'] = params['input_dim_stage2']  # Ex. TAO: 7  # (H_2, A_2)

            start_time = time.time()  # Start time recording
            trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS_u, config_number)
            end_time = time.time()  # End time recording
            print(f"Total time taken to run surr_opt: { end_time - start_time} seconds")
            # Append epoch model results from surr_opt
            epoch_num_model_lst.append(epoch_num_model_DS)
            # Store losses 
            losses_dict['DS'][replication] = trn_val_loss_tpl_DS
            
        # eval_DTR 
        print("Evaluation started") 
        start_time = time.time()  # Start time recording 
        # V_replications, df_DQL, df_DS, df_Tao = eval_DTR(V_replications, replication, df_DQL, df_DS, df_Tao, params_DQL_f, params_DS_f, config_number)
        V_replications, df_DQL, df_DS, df_Tao = eval_DTR(V_replications, replication, df_DQL, df_DS, df_Tao, params_DQL_u, params_DS_u, config_number)
        end_time = time.time()  # End time recording
        print(f"Total time taken to run eval_DTR: { end_time - start_time} seconds \n\n")
                
    return V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst


def run_training(config, config_fixed, config_updates, V_replications, config_number, replication_seed):
    torch.manual_seed(replication_seed)
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = simulations(V_replications, local_config, config_fixed, config_number)
    
    if not any(V_replications[key] for key in V_replications):
        warnings.warn("V_replications is empty. Skipping accuracy calculation.")
    else:
        VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh = extract_value_functions_separate(V_replications)
        return VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst
 

# parallelized 

def run_training_with_params(params):

    config, config_fixed, current_config, V_replications, i, config_number = params
    return run_training(config, config_fixed, current_config, V_replications, config_number, replication_seed=i)
 


        
def run_grid_search(config, config_fixed, param_grid):
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
                params = (config, config_fixed, current_config, V_replications, i, config_number)
                future = executor.submit(run_training_with_params, params)
                future_to_params[future] = (current_config, i)

        for future in concurrent.futures.as_completed(future_to_params):
            current_config, i = future_to_params[future]            
            performance_DQL, performance_DS, performance_Tao, performance_Beh, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst = future.result()
            
            print(f'Configuration {current_config}, replication {i} completed successfully.')
            
            # Processing performance DataFrame for both methods
            performances_DQL = pd.DataFrame()
            performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

            performances_DS = pd.DataFrame()
            performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            performances_Tao = pd.DataFrame()
            performances_Tao = pd.concat([performances_Tao, performance_Tao], axis=0)

            performances_Beh = pd.DataFrame()
            performances_Beh = pd.concat([performance_Beh, performance_Beh], axis=0)

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

            # Calculating the standard deviation for "Method's Value fn."
            performance_DQL_std = performances_DQL["Method's Value fn."].std()
            performance_DS_std = performances_DS["Method's Value fn."].std()
            performance_Tao_std = performances_Tao["Method's Value fn."].std()
            performance_Beh_std = performances_Beh["Method's Value fn."].std()

            # Check if the configuration key exists in the results dictionary
            if config_key not in results:
                # If not, initialize it with dictionaries for each model containing the mean values
                results[config_key] = {
                    'DQL': {"Method's Value fn.": performance_DQL_mean, 
                            "Method's Value fn. SD": performance_DQL_std, 
                            },
                    'DS': {"Method's Value fn.": performance_DS_mean, 
                           "Method's Value fn. SD": performance_DS_std,
                           },
                    'Tao': {"Method's Value fn.": performance_Tao_mean, 
                           "Method's Value fn. SD": performance_Tao_std,
                           },
                    'Behavioral': {"Method's Value fn.": performance_Beh_mean, 
                           "Method's Value fn. SD": performance_Beh_std,
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

    config_fixed = copy.deepcopy(config)

    # # Define parameter grid for grid search
    # param_grid = {
    #     'activation_function': [ 'none'], # elu, relu, sigmoid, tanh, leakyrelu, none
    #     'batch_size': [10, 25, 200], # 50
    #     'optimizer_lr': [0.07], # 0.1, 0.01, 0.07, 0.001
    #     'num_layers': [2, 4], # 1,2,3,4,5,6,7
    #     # 'n_epoch':[60, 150],
    #     # "surrogate_num": 1  
    # }
    
    # Tao case
    # param_grid = {
    #     'activation_function': ['none'], # elu, relu, sigmoid, tanh, leakyrelu, none
    #     'learning_rate': [0.07],
    #     'num_layers': [2, 4],
    #     'batch_size': [500],
    #     'hidden_dim_stage1': [40],  # Number of neurons in the hidden layer of stage 1
    #     'hidden_dim_stage2': [10],  # Number of neurons in the hidden layer of stage 2 
    #     'dropout_rate': [0],  # Dropout rate to prevent overfitting
    #     'n_epoch': [250]
    # }

    # Scheme 5 case
    param_grid = {
        'activation_function': ['none'], # 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'
        'learning_rate': [0.07],
        'num_layers': [1], # 2, 4 
        'batch_size': [100],
        'hidden_dim_stage1': [10],  #40,  Number of neurons in the hidden layer of stage 1
        'hidden_dim_stage2': [10],  #10,  Number of neurons in the hidden layer of stage 2 
        'dropout_rate': [0.4],  # 0 Dropout rate to prevent overfitting
        'n_epoch': [300], #60, 70, 90, 250 
        # 'neu':[2,4,10] 
    }

    # # Scheme 5 case
    # param_grid = {
    #     'activation_function': ['relu'], # 'elu', 'relu', 'leakyrelu', 'none', 'sigmoid', 'tanh'
    #     'learning_rate': [0.07],
    #     'num_layers': [3], # 4
    #     'batch_size': [1000],
    #     'hidden_dim_stage1': [10],  # Number of neurons in the hidden layer of stage 1
    #     'hidden_dim_stage2': [10],  # Number of neurons in the hidden layer of stage 2 
    #     'dropout_rate': [0],  # Dropout rate to prevent overfitting
    #     'n_epoch': [70] #60, 90, 250
    # }

    # Perform operations whose output should go to the file
    run_grid_search(config, config_fixed, param_grid)
    

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


    
