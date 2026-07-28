import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)

from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from utils.BasicResNet import BasicResNet
import utils.config as config

from utils.architecture2 import Architecture

class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, inp, out, mu, std):
        self.inp = torch.from_numpy(inp).float()
        self.out = torch.from_numpy(out).float()

        self.mu = torch.from_numpy(mu).float()
        self.std = torch.from_numpy(std).float()

    def __len__(self):
        return self.inp.size(0)

    def __getitem__(self, idx):
        x = (self.inp[idx] - self.mu) / self.std
        y = self.out[idx]
        return x, y




import argparse

def gen_dope_data():
    max_arch = Architecture(4, 8, 128, 2, 3, 255, 255, 1, 4, 5, 255, 2, 6, 7, 255)
    max_vals = max_arch.getModelVector() 
    max_vals.extend([1.0, 1.0, 1.0])
    return np.array(max_vals)

def remove_outliers(data_array):
    """
    Removes rows from a numpy array where the value in the last column is higher than 1.

    Parameters:
    data_array (np.ndarray): The input numpy array.

    Returns:
    np.ndarray: A new numpy array with rows removed where the last column value is higher than 1.
    """
    # Get the last column
    last_column = data_array[:, -2]
    
    # Create a boolean mask where the last column value is less than or equal to 1
    mask = last_column < 1
    
    # Use the mask to filter the rows
    filtered_array = data_array[mask]
    
    return filtered_array



# Use GPU if available
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


dataset = np.load("data/train_data.npy").astype(np.float32)
val_dataset = np.load("data/val_data.npy").astype(np.float32)
train_set = dataset
val_set = val_dataset

print(train_set.shape)
train_set = remove_outliers(train_set)
print(train_set.shape)



out_column = 3  # You can set this variable to 3, 2, or 1 to select the appropriate columns

# Train & Validation dataset
inp = train_set[:, :-3].astype(np.float32)  # Batch x (No. of features - 3)
out = train_set[:, -2].astype(np.float32)  # Batch x 2, taking the specified columns

# Validation Data
inp_val = val_set[:, :-3].astype(np.float32)  # Batch x (No. of features - 3)
out_val = val_set[:, -2].astype(np.float32)  # Batch x 2, taking the specified columns


# Normalization Constants
constant = 1e-9
mu, std = np.mean(inp, axis=0), np.maximum(np.std(inp, axis=0), constant)

#np.savetxt("synthesis_mu.txt", mu, delimiter = ',')
#np.savetxt("synthesis_std.txt", std, delimiter = ',')
#
# Batch Size 
batch_size = 32 * 1
val_batch_size = 256

# Using PyTorch Dataloader
train_dataset = ChipDataset(inp, out, mu, std)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
 
# Validation Dataloader
val_dataset =ChipDataset(inp_val, out_val, mu, std)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=0, drop_last=True)


# Based on optuna search
num_features = 190
depth = 3
weight_decay = 5.735733391055444e-05
learning_rate = 0.0006399856935907461
epochs = 130


# Initializing BasicResNet
model = BasicResNet(config.SYNTHESIS_NUM_INPUTS,
                    num_features,
                    depth,
                    1,
                    config.SYNTHESIS_NORMALIZATION,
                    config.SYNTHESIS_ACTIVATION).to(device)

# Training Function
def train(epoch, scheduler):
    model.train()
    train_loss = 0
    dataloader = train_loader 
    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device).view(-1)
        optimizer.zero_grad()
        y_pred, _ = model(x)
        y_pred = y_pred.view(-1)
        loss = criterion(y_pred, y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    scheduler.step()

# Define the parameters that should and should not undergo weight decay
decay = []
no_decay = []
for name, param in model.named_parameters():
    if 'weight' in name and 'LayerNorm' not in name:
        decay.append(param)
    else:
        no_decay.append(param)

# Optimizer
lr = learning_rate
weight_decay = weight_decay
optimizer = optim.Adam([
    {'params': decay, 'weight_decay': weight_decay},
    {'params': no_decay, 'weight_decay': 0.0}], lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
criterion = nn.MSELoss()

# No. of training cycles over the epochs
from sklearn.metrics import mean_squared_error

def get_top_1_percent(values):
    # Convert the list to a numpy array (useful for percentile operations)
    values = np.array(values)
    
    # Calculate the index for the top 1%
    top_1_percent_index = int(np.ceil(len(values) * 0.01))
    
    # Sort the values in descending order
    sorted_values = np.sort(values)[::-1]  # Sort in descending order
    
    # Get the top 1% values
    top_1_percent_values = sorted_values[:top_1_percent_index]
    
    return top_1_percent_values

last_errors = []
# Training Loop
for epoch in range(1, epochs + 1):
    train(epoch, scheduler)
    
    errors = []
    mae_list = []
    median_list = []
    max_error_list = []
    top_1_percent_list = []
    mse_list = []
    rel_error_list = []


    preds = []
    truths = []

    for (v_inp, v_out) in val_loader:

        # Input and Output
        v_inp = v_inp.to(device)
        v_out = v_out.to(device)

        # Forward Pass
        with torch.no_grad():
            pred = model(v_inp)
            predicted, _ = pred  # Flatten predictions
            predicted = predicted.view(-1)
            ground_truth = v_out.view(-1)

            preds.extend(predicted.tolist())
            truths.extend(ground_truth.tolist())
                        
            # Mean Squared Error (MSE)

            # Normalized error for the initial print statement
            err = abs(ground_truth - predicted) / ground_truth
            errors.extend(err.tolist())
    
    # Calculate mean of statistics for the current epoch
    median_error = np.median(errors)
    top_1_percent_error = np.mean(get_top_1_percent(errors))

    # Calculate MSE
    mse = mean_squared_error(truths, preds)

    print(f"epoch: {epoch-1:4d} | "
      f"error: {sum(errors)/len(errors):.3f} | "
      f"max: {max(errors):.3f} | "
      f"Median Error: {median_error:.4f} | "
      f"Top 1% Mean Error: {top_1_percent_error:.4f} | "
      f"MSE: {mse:.4f}")

    last_errors = errors
    

torch.save(model.state_dict(), "area_model.pth")
