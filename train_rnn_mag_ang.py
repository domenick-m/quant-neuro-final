#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import os
import yaml
import torch
import wandb
import shutil
import argparse
import numpy as np
import torch.nn as nn
from joblib import dump
from utils import prepare_datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torcheval.metrics.functional import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from rnn_model import RNNModel


#%% ----------------------------------------------------------------------------
# Parse command line arguments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--add', type=str, default=None, help='Add sweep agent to specified sweep ID.')
parser.add_argument('--gpu', type=int, default=None, help='GPU to add agent to.')
args = parser.parse_args()


#%% ----------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
# Which GPU should we train on (if available)
GPU = 1 if args.gpu is None else args.gpu
NUM_RUNS = 100

PRED_THETA = True

# Load config parameters from file
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Log all runs to this wandb project
wand_proj_name = f'{config["WANDB_BASE_PROJECT"]}_rnn_mag_ang_theta'

# Create folder to store runs locally if it doesn't exist
runs_folder = './wandb_runs/rnn_mag_ang'
if not os.path.exists(runs_folder):
    os.makedirs(runs_folder)

# Prepare datasets (extract, smooth, lag, etc...)
trainval, test = prepare_datasets(config)
trainval_spikes, trainval_behavior = trainval

# Convert behavior to magnitude and angle
x_vel, y_vel = np.split(trainval_behavior, 2, axis=-1)
magnitude = np.sqrt(x_vel ** 2 + y_vel ** 2)
angle_rads = np.arctan2(y_vel, x_vel)
trainval_behavior = np.concatenate([magnitude, angle_rads], axis=-1)

# split into training and validation sets
tv_splits = train_test_split(trainval_spikes, 
                             trainval_behavior, 
                             test_size=0.2, 
                             random_state=1)
train_spikes, val_spikes, train_behavior, val_behavior = tv_splits

train_spikes_tensor = torch.Tensor(train_spikes)
train_behavior_tensor = torch.Tensor(train_behavior)
val_spikes_tensor = torch.Tensor(val_spikes)
val_behavior_tensor = torch.Tensor(val_behavior)

n_neurons = train_spikes_tensor.shape[-1] # rnn input size
train_dataset = TensorDataset(train_spikes_tensor, train_behavior_tensor)
val_dataset = TensorDataset(val_spikes_tensor, val_behavior_tensor)


#%% -----------------------------------
# Set up the W&B Sweep config 
# -------------------------------------
sweep_config = {
    "name": "rnn_mag_ang_random_sweep",
    "method": "random",
    "parameters": {
        "lr": {"values": config["RNN_LR"]},
        "dropout": {"values": config["RNN_DROPOUT"]},
        "n_layers": {"values": config["RNN_LAYERS"]},
        "n_epochs": {"values": config["RNN_EPOCHS"]},
        "batch_size": {"values": config["RNN_BATCH_SIZE"]},
        "hidden_size": {"values": config["RNN_HIDDEN_SIZE"]}
    }
}


# ------------------------------------------------------------------------------
# Function to train GRU using random search over HPs
# ------------------------------------------------------------------------------
def train():
    device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize a new wandb run
    wandb.init()
    
    # Extract this runs hyperparameters
    lr = wandb.config.lr
    dropout = wandb.config.dropout
    n_layers = wandb.config.n_layers
    n_epochs = wandb.config.n_epochs
    batch_size = wandb.config.batch_size
    hidden_size = wandb.config.hidden_size

    # Give the run a new name to reflect the hyperparameters
    wandb.run.name = f'lr_{lr}_drop_{dropout}_layers_{n_layers}_hsz_{hidden_size}'

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

    # Set up the RNN models with the current configurations
    mag_model = RNNModel(input_size=n_neurons, 
                         hidden_size=hidden_size,
                         num_layers=n_layers,
                         dropout=dropout,
                         n_outputs=1)
    ang_model = RNNModel(input_size=n_neurons, 
                         hidden_size=hidden_size,
                         num_layers=n_layers,
                         dropout=dropout,
                         n_outputs=1 if PRED_THETA else 2)
    mag_model = mag_model.to(device)
    ang_model = ang_model.to(device)

    # Set up loss and optimizer
    mse_loss = nn.MSELoss()
    mag_optimizer = torch.optim.AdamW(mag_model.parameters(), lr=lr)
    ang_optimizer = torch.optim.AdamW(ang_model.parameters(), lr=lr)

    # Save a checkpoint at the best validation loss
    best_mag_val_loss = float('inf')
    best_ang_val_loss = float('inf')

    # Training loop
    for epoch in range(n_epochs):
        print(f' Epoch: {epoch + 1} / {n_epochs}{" " * 20}', end='\r')
        mag_model.train()
        ang_model.train()
        mag_train_loss, ang_train_loss, train_r2 = 0, 0, 0
        for spikes, behavior in train_dl:
            # Forward pass through model
            spikes = spikes.to(device)
            pred_mag = mag_model(spikes)
            pred_ang = ang_model(spikes)

            # Calculate magnitude loss            
            true_mag, true_ang = torch.split(behavior.to(device), 1, dim=-1)
            mag_loss = mse_loss(pred_mag, true_mag)

            # Calculate anglular loss
            true_sin, true_cos = torch.sin(true_ang), torch.cos(true_ang)
            pred_sin, pred_cos = (torch.sin(pred_ang), torch.cos(pred_ang)) if PRED_THETA else \
                                 torch.split(pred_ang, 1, dim=-1) 
            sin_loss = mse_loss(pred_sin, true_sin)
            cos_loss = mse_loss(pred_cos, true_cos)
            ang_loss = sin_loss + cos_loss
            
            # Calculate R^2
            pred_vel = torch.cat([pred_cos, pred_sin], axis=-1) * pred_mag
            true_vel = torch.cat([true_cos, true_sin], axis=-1) * true_mag
            r2 = r2_score(pred_vel.reshape((-1, 2)), true_vel.reshape((-1, 2)))

            # Update epoch metrics
            mag_train_loss += mag_loss.cpu().detach().numpy()
            ang_train_loss += ang_loss.cpu().detach().numpy()
            train_r2 += r2.cpu().detach().numpy()

            # Backpropagate and update weights
            mag_loss.backward()
            mag_optimizer.step()
            mag_optimizer.zero_grad(set_to_none=True)
            ang_loss.backward()
            ang_optimizer.step()
            ang_optimizer.zero_grad(set_to_none=True)

        # Log the training loss to Weights and Biases
        wandb.log({"mag_train_loss": mag_train_loss / len(train_dl), 
                   "ang_train_loss": ang_train_loss / len(train_dl),
                   "train_r2": train_r2 / len(train_dl),
                   "epoch": epoch})
        
        # Validation loop
        mag_model.eval()
        ang_model.eval()
        mag_val_loss, ang_val_loss, val_r2 = 0, 0, 0
        with torch.no_grad():
            for spikes, behavior in val_dl:
                # Forward pass through model
                spikes = spikes.to(device)
                pred_mag = mag_model(spikes)
                pred_ang = ang_model(spikes)

                # Calculate magnitude loss            
                true_mag, true_ang = torch.split(behavior.to(device), 1, dim=-1)
                mag_loss = mse_loss(pred_mag, true_mag)

                # Calculate anglular loss
                true_sin, true_cos = torch.sin(true_ang), torch.cos(true_ang)
                pred_sin, pred_cos = (torch.sin(pred_ang), torch.cos(pred_ang)) if PRED_THETA else \
                                    torch.split(pred_ang, 1, dim=-1) 
                sin_loss = mse_loss(pred_sin, true_sin)
                cos_loss = mse_loss(pred_cos, true_cos)
                ang_loss = sin_loss + cos_loss
                
                # Calculate R^2
                pred_vel = torch.cat([pred_cos, pred_sin], axis=-1) * pred_mag
                true_vel = torch.cat([true_cos, true_sin], axis=-1) * true_mag
                r2 = r2_score(pred_vel.reshape((-1, 2)), true_vel.reshape((-1, 2)))

                # Update epoch metrics
                mag_val_loss += mag_loss.cpu().numpy()
                ang_val_loss += ang_loss.cpu().numpy()
                val_r2 += r2.cpu().numpy()

        # Log the validation loss to Weights and Biases
        avg_mag_val_loss = mag_val_loss / len(val_dl)
        avg_ang_val_loss = ang_val_loss / len(val_dl)
        wandb.log({"mag_val_loss": avg_mag_val_loss, 
                   "ang_val_loss": avg_ang_val_loss, 
                   "val_r2": val_r2 / len(val_dl),
                   "epoch": epoch})
        
        # Save a checkpoint if this is the best validation loss
        if avg_mag_val_loss < best_mag_val_loss:
            best_mag_val_loss = avg_mag_val_loss
            torch.save(mag_model.state_dict(), os.path.join(runs_folder, f'{wandb.run.name}-mag-best.pth'))
        if avg_ang_val_loss < best_ang_val_loss:
            best_ang_val_loss = avg_ang_val_loss
            torch.save(ang_model.state_dict(), os.path.join(runs_folder, f'{wandb.run.name}-ang-best.pth'))

    # Save the trained models
    torch.save(mag_model.state_dict(), os.path.join(runs_folder, f'{wandb.run.name}-mag-last.pth'))
    torch.save(ang_model.state_dict(), os.path.join(runs_folder, f'{wandb.run.name}-ang-last.pth'))

    # Close wandb run
    wandb.finish()


# ------------------------------------------------------------------------------
# Start random search sweep and init an agent
# ------------------------------------------------------------------------------
if args.add is None:
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, 
                        project=wand_proj_name, 
                        entity=config["WANDB_ENTITY"])
    # Start a sweep agent
    wandb.agent(sweep_id, function=train, count=NUM_RUNS)
else:
    # Add an agent to the sweep
    wandb.agent(args.add, 
                function=train, 
                count=NUM_RUNS, 
                project=wand_proj_name, 
                entity=config["WANDB_ENTITY"])


# ------------------------------------------------------------------------------
# Get the best model from the sweep and rename / copy it
# ------------------------------------------------------------------------------
api = wandb.Api()
sweep = api.sweep(f'{config["WANDB_ENTITY"]}/{wand_proj_name}/{sweep_id}')
lowest_mag_val_loss = float('inf')
lowest_ang_val_loss = float('inf')
best_mag_run_name = None
best_ang_run_name = None

# Find the run with the lowest validation loss
for run in sweep.runs:
    mag_val_loss = run.summary.get('mag_val_loss', None)
    ang_val_loss = run.summary.get('ang_val_loss', None)
    if mag_val_loss is not None and mag_val_loss < lowest_mag_val_loss:
        lowest_mag_val_loss = mag_val_loss
        best_mag_run_name = run.name
    if ang_val_loss is not None and ang_val_loss < lowest_ang_val_loss:
        lowest_ang_val_loss = ang_val_loss
        best_ang_run_name = run.name

# Copy the best model from the runs folder
best_mag_model_path = os.path.join(runs_folder, f'{best_mag_run_name}-mag-best.pth')
best_ang_model_path = os.path.join(runs_folder, f'{best_ang_run_name}-ang-best.pth')
shutil.copy(best_mag_model_path, f'rnn_model_mag_{PRED_THETA}.pth')
shutil.copy(best_ang_model_path, f'rnn_model_ang_{PRED_THETA}.pth')