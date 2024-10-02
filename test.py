# Basic implementation


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
n_samples = 1000  # Number of samples
p = 5             # Number of covariates
sigma = 0.1       # Standard deviation of noise

# Generate covariates X ~ N(0, I_p)
X = np.random.randn(n_samples, p)

# Define treatment effect vectors for each stage and treatment
beta_1 = {
    1: np.array([1, 0, 0, 0, 0]),
    2: np.array([0, 1, 0, 0, 0]),
    3: np.array([0, 0, 1, 0, 0])
}

beta_2 = {
    1: np.array([1, 0, 0, 0, 0]),
    2: np.array([0, 1, 0, 0, 0]),
    3: np.array([0, 0, 1, 0, 0])
}

# State transition parameters
gamma = {
    1: np.array([0.5, 0, 0, 0, 0]),
    2: np.array([0, 0.5, 0, 0, 0]),
    3: np.array([0, 0, 0.5, 0, 0])
}

# Behavioral policy (uniform random)
def behavioral_policy():
    return np.random.choice([1, 2, 3])

# Initialize lists to store data
O1_list = []
A1_list = []
Y1_list = []
O2_list = []
A2_list = []
Y2_list = []

for i in range(n_samples):
    # Stage 1
    O1 = X[i]  # O1 = X
    H1 = O1    # History at stage 1
    A1 = behavioral_policy()
    epsilon1 = np.random.normal(0, sigma)
    Y1 = O1 @ beta_1[A1] + epsilon1

    # State transition to stage 2
    eta = np.random.normal(0, sigma, size=p)
    O2 = O1 + gamma[A1] + eta

    # Stage 2
    H2 = np.concatenate([H1, [A1, Y1], O2])  # History at stage 2
    A2 = behavioral_policy()
    epsilon2 = np.random.normal(0, sigma)
    Y2 = O2 @ beta_2[A2] + epsilon2

    # Store data
    O1_list.append(O1)
    A1_list.append(A1)
    Y1_list.append(Y1)
    O2_list.append(O2)
    A2_list.append(A2)
    Y2_list.append(Y2)

# Convert lists to numpy arrays
O1 = np.array(O1_list)
A1 = np.array(A1_list)
Y1 = np.array(Y1_list)
O2 = np.array(O2_list)
A2 = np.array(A2_list)
Y2 = np.array(Y2_list)



# Prepare data
# Adjust outcomes to be non-negative
Y1_min = Y1.min()
Y2_min = Y2.min()
if Y1_min < 0:
    Y1 = Y1 - Y1_min
if Y2_min < 0:
    Y2 = Y2 - Y2_min

# Define histories
H1 = O1  # Already numpy array of shape (n_samples, p)
H2 = np.hstack([H1, A1.reshape(-1, 1), Y1.reshape(-1, 1), O2])

# Convert data to PyTorch tensors
H1_tensor = torch.from_numpy(H1).float()
A1_tensor = torch.from_numpy(A1 - 1).long()  # Actions are 0, 1, 2
Y1_tensor = torch.from_numpy(Y1).float()
H2_tensor = torch.from_numpy(H2).float()
A2_tensor = torch.from_numpy(A2 - 1).long()  # Actions are 0, 1, 2
Y2_tensor = torch.from_numpy(Y2).float()

# Define the surrogate function phi (sigmoid function)
phi = torch.sigmoid

# Define neural network models for g_{11}, g_{12}, g_{21}, g_{22}
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)  # Output is scalar g_{tj}

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize models
g_11 = Net(p)
g_12 = Net(p)
g_21 = Net(H2_tensor.shape[1])
g_22 = Net(H2_tensor.shape[1])

# Define optimizer
params = list(g_11.parameters()) + list(g_12.parameters()) + list(g_21.parameters()) + list(g_22.parameters())
optimizer = optim.Adam(params, lr=0.001)

# Behavioral policy probabilities (since policy is uniform random, probability is 1/3)
pi1 = torch.full((n_samples,), 1/3)
pi2 = torch.full((n_samples,), 1/3)

# Define the Gamma function
def Gamma(a, b, A):
    """
    a, b: Tensors of shape (n_samples,)
    A: Tensor of shape (n_samples,) with values 0, 1, or 2 corresponding to actions 1, 2, 3
    """
    phi_a = phi(a)
    phi_b = phi(b)
    phi_ba = phi(b - a)
    phi_ab = phi(a - b)
    phi_neg_a = phi(-a)
    phi_neg_b = phi(-b)
    Gamma_val = torch.zeros_like(a)
    # When A == 0 (action 1)
    idx = (A == 0)
    Gamma_val[idx] = phi_a[idx] * phi_b[idx]
    # When A == 1 (action 2)
    idx = (A == 1)
    Gamma_val[idx] = phi_ba[idx] * phi_neg_a[idx]
    # When A == 2 (action 3)
    idx = (A == 2)
    Gamma_val[idx] = phi_ab[idx] * phi_neg_b[idx]
    return Gamma_val

# Define data-dependent weights kappa_i
kappa = (Y1_tensor + Y2_tensor) / (pi1 * pi2)

# Training loop
n_epochs = 1000  # Adjust as needed
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Compute g_{tj}(H_t)
    g11 = g_11(H1_tensor).squeeze()  # Shape: (n_samples,)
    g12 = g_12(H1_tensor).squeeze()
    g21 = g_21(H2_tensor).squeeze()
    g22 = g_22(H2_tensor).squeeze()
    
    # Compute f_{tj}(H_t)
    f_t1_stage1 = torch.zeros_like(g11)  # f_{11} = 0
    f_t2_stage1 = -g11                  # f_{12} = -g_{11}
    f_t3_stage1 = -g12                  # f_{13} = -g_{12}
    
    f_t1_stage2 = torch.zeros_like(g21)  # f_{21} = 0
    f_t2_stage2 = -g21                   # f_{22} = -g_{21}
    f_t3_stage2 = -g22                   # f_{23} = -g_{22}
    
    # Compute Gamma functions
    Gamma1 = Gamma(f_t2_stage1, f_t3_stage1, A1_tensor)
    Gamma2 = Gamma(f_t2_stage2, f_t3_stage2, A2_tensor)
    
    # Compute surrogate loss (we want to maximize it)
    V_hat = torch.mean(kappa * Gamma1 * Gamma2)
    loss = -V_hat  # Negative because we use optimizers that minimize loss
    
    # Backward pass and optimization step
    loss.backward()
    optimizer.step()
    
    # Optionally print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# Extract estimated policies
with torch.no_grad():
    # Stage 1
    g11_vals = g_11(H1_tensor).squeeze()
    g12_vals = g_12(H1_tensor).squeeze()
    f_t1_stage1 = torch.zeros_like(g11_vals)  # f_{11} = 0
    f_t2_stage1 = -g11_vals                  # f_{12} = -g_{11}
    f_t3_stage1 = -g12_vals                  # f_{13} = -g_{12}
    f_vals_stage1 = torch.stack([f_t1_stage1, f_t2_stage1, f_t3_stage1], dim=1)  # Shape: (n_samples, 3)
    d1_hat = torch.argmax(f_vals_stage1, dim=1)  # Estimated optimal actions at stage 1 (0, 1, or 2)

    # Stage 2
    g21_vals = g_21(H2_tensor).squeeze()
    g22_vals = g_22(H2_tensor).squeeze()
    f_t1_stage2 = torch.zeros_like(g21_vals)  # f_{21} = 0
    f_t2_stage2 = -g21_vals                   # f_{22} = -g_{21}
    f_t3_stage2 = -g22_vals                   # f_{23} = -g_{22}
    f_vals_stage2 = torch.stack([f_t1_stage2, f_t2_stage2, f_t3_stage2], dim=1)  # Shape: (n_samples, 3)
    d2_hat = torch.argmax(f_vals_stage2, dim=1)  # Estimated optimal actions at stage 2 (0, 1, or 2)

# Define true optimal policies (since we know the DGP)
# Optimal action is the one corresponding to the largest component in O_t

# Stage 1 optimal actions
O1_tensor = torch.from_numpy(O1).float()
optimal_A1 = torch.argmax(O1_tensor[:, :3], dim=1)  # Actions 0, 1, or 2

# Stage 2 optimal actions
O2_tensor = torch.from_numpy(O2).float()
optimal_A2 = torch.argmax(O2_tensor[:, :3], dim=1)  # Actions 0, 1, or 2

# Compute accuracy
accuracy_stage1 = (d1_hat == optimal_A1).float().mean().item()
accuracy_stage2 = (d2_hat == optimal_A2).float().mean().item()
print(f'Stage 1 Accuracy: {accuracy_stage1:.4f}')
print(f'Stage 2 Accuracy: {accuracy_stage2:.4f}')

# Estimate the value function by simulating trajectories under the estimated policies
def simulate_value_function(n_simulations):
    total_rewards = []
    for _ in range(n_simulations):
        rewards = []
        for i in range(n_samples):
            # Stage 1
            O1_i = X[i]
            A1_i = d1_hat[i].item() + 1  # Convert back to actions 1, 2, 3
            epsilon1 = np.random.normal(0, sigma)
            Y1_i = O1_i @ beta_1[A1_i] + epsilon1

            # State transition
            eta = np.random.normal(0, sigma, size=p)
            O2_i = O1_i + gamma[A1_i] + eta

            # Stage 2
            H2_i = np.concatenate([O1_i, [A1_i, Y1_i], O2_i])
            A2_i = d2_hat[i].item() + 1  # Convert back to actions 1, 2, 3
            epsilon2 = np.random.normal(0, sigma)
            Y2_i = O2_i @ beta_2[A2_i] + epsilon2

            total_reward = Y1_i + Y2_i
            rewards.append(total_reward)
        total_rewards.append(np.mean(rewards))
    return np.mean(total_rewards), np.std(total_rewards)

# Simulate the value function under the estimated policies
estimated_value_mean, estimated_value_std = simulate_value_function(n_simulations=100)
print(f'Estimated Value Function: Mean = {estimated_value_mean:.4f}, Std = {estimated_value_std:.4f}')

# Compute the true optimal value function
def simulate_optimal_value_function(n_simulations):
    total_rewards = []
    for _ in range(n_simulations):
        rewards = []
        for i in range(n_samples):
            # Stage 1
            O1_i = X[i]
            A1_i = optimal_A1[i].item() + 1  # Convert back to actions 1, 2, 3
            epsilon1 = np.random.normal(0, sigma)
            Y1_i = O1_i @ beta_1[A1_i] + epsilon1

            # State transition
            eta = np.random.normal(0, sigma, size=p)
            O2_i = O1_i + gamma[A1_i] + eta

            # Stage 2
            H2_i = np.concatenate([O1_i, [A1_i, Y1_i], O2_i])
            A2_i = optimal_A2[i].item() + 1  # Convert back to actions 1, 2, 3
            epsilon2 = np.random.normal(0, sigma)
            Y2_i = O2_i @ beta_2[A2_i] + epsilon2

            total_reward = Y1_i + Y2_i
            rewards.append(total_reward)
        total_rewards.append(np.mean(rewards))
    return np.mean(total_rewards), np.std(total_rewards)

# Simulate the value function under the true optimal policies
optimal_value_mean, optimal_value_std = simulate_optimal_value_function(n_simulations=100)
print(f'True Optimal Value Function: Mean = {optimal_value_mean:.4f}, Std = {optimal_value_std:.4f}')

# Compute the value difference
value_difference = optimal_value_mean - estimated_value_mean
print(f'Value Difference (Optimal - Estimated): {value_difference:.4f}')
