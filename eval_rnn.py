#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from nlb_tools.evaluation import evaluate
from sklearn.model_selection import KFold
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from rnnmodel import RNNMODEL


#%% ----------------------------------------------------------------------------
# Static Parameters
# ------------------------------------------------------------------------------
BINSIZE = 5 # ms - bin size for resampling
LAG = 100 # ms - how much to delay the kinematics by
SMOOTH_SD = 40 # ms - std of gaussian kernel for smoothing
DATASET = 'mc_maze' # NLB name of dataset

WANDB_PROJECT = 'quantNeuroTest' # name of wandb project
WANDB_ENTITY = 'domenick-m' # wandb username (change to yours)


#%% ----------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------
RNN_LAYERS = 1
RNN_HIDDEN_SIZE = 64
RNN_DROPOUT = 0.3 
RNN_LR = 0.001 
RNN_EPOCHS = 1000
RNN_BATCH_SIZE = 32


#%% ----------------------------------------------------------------------------
# Dataset Set Up
# ------------------------------------------------------------------------------
DATASET_IDS = {
    'mc_maze': 128,
    'mc_rtt': 129,
    'mc_maze_large': 138,
    'mc_maze_medium': 139,
    'mc_maze_small': 140,
}
DATASET
id = DATASET_IDS[DATASET]
# get the path to the NLB data using the dataset ID
data_folder = Path.cwd() / 'data' / f'000{id}'

# download dataset if it doesn't exist
if not data_folder.exists():
    print(f'Dataset "{DATASET}" not downloaded! Downloading now.')
    os.system(f'mkdir {data_folder}')
    os.system(f'dandi download https://dandiarchive.org/dandiset/000{id} -o data')

# get the path to the folder inside the data folder
data_path = next((p for p in data_folder.iterdir() if p.is_dir()), None)


#%% ----------------------------------------------------------------------------
# Prepare / Preprocess Data
# ------------------------------------------------------------------------------
# Warning! This step takes ~1.5 mins
dataset = NWBDataset(data_path) # load dataset from file
dataset.resample(BINSIZE) # resample to 5 ms bins

# extract training and validation spikes
trainval_dict = make_train_input_tensors(dataset, DATASET, save_file=False) 
behav_dict = make_eval_target_tensors(dataset, DATASET, save_file=False)

# combine heldin and heldout neurons
stacked = [trainval_dict[f'train_spikes_{i}'] for i in ['heldin', 'heldout']]
trainval_spikes = np.concatenate(stacked, axis=2)

# smooth spikes using gaussian window
kern_sd = int(round(SMOOTH_SD / dataset.bin_width))
window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
filt = lambda x: np.convolve(x, window / np.sum(window), 'same')
trainval_smth_spikes = np.apply_along_axis(filt, 1, trainval_spikes)

# extract training and validation kinematics
trainval_behavior = behav_dict[DATASET]['train_behavior']

# delay kinematics by LAG ms
lag_bins = int(LAG / BINSIZE) # number of bins to delay kinematics by
trainval_smth_spikes = trainval_smth_spikes[:, :-lag_bins]
trainval_behavior = trainval_behavior[:, lag_bins:]

# split into training and validation sets
tv_splits = train_test_split(trainval_smth_spikes, 
                             trainval_behavior, 
                             test_size=0.2, 
                             random_state=1)
train_spikes, val_spikes, train_behavior, val_behavior = tv_splits


#%% ----------------------------------------------------------------------------
# Prepare data for training 
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_spikes_tensor = torch.Tensor(train_spikes, device=device)
val_spikes_tensor = torch.Tensor(val_spikes, device=device)

train_behavior_tensor = torch.Tensor(train_behavior, device=device)
val_behavior_tensor = torch.Tensor(val_behavior, device=device)

input_size = train_spikes_tensor.size()[-1] # Specify the input size based on your data
output_size = train_behavior_tensor.size()[-1]

train_dataset = TensorDataset(train_spikes_tensor, train_behavior_tensor)
val_dataset = TensorDataset(val_spikes_tensor, val_behavior_tensor)
train_loader = DataLoader(train_dataset, batch_size=RNN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=RNN_BATCH_SIZE, shuffle=False)  


#%% -----------------------------------
# Run a single run
# -------------------------------------
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f'RNN_test')

# Set up the RNN model, optimizer, and loss function
model = RNNMODEL(input_size=input_size, 
                 num_layers=RNN_LAYERS, 
                 hidden_size=RNN_HIDDEN_SIZE, 
                 dropout=RNN_DROPOUT).to(device)
optimizer = optim.AdamW(model.parameters(), lr=RNN_LR)
mse_loss = nn.MSELoss()

# Training loop
for epoch in range(RNN_EPOCHS):
    model.train()
    total_loss, total_r2 = 0, 0
    for batch_idx, (spikes, behavior) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(spikes)
        loss = mse_loss(output, behavior)
        r2 = r2_score(output.reshape((-1, output_size)).detach().cpu().numpy(), 
                      behavior.reshape((-1, output_size)).detach().cpu().numpy(), 
                      multioutput='uniform_average')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_r2 += r2.item()

    # Validation loop
    model.eval()
    val_loss, val_r2 = 0, 0
    with torch.no_grad():
        for batch_idx, (spikes, behavior) in enumerate(val_loader):
            spikes, behavior = spikes.to(device), behavior.to(device)
            output = model(spikes)
            loss = mse_loss(output, behavior)
            r2 = r2_score(output.reshape((-1, output_size)), 
                          behavior.reshape((-1, output_size)), 
                          multioutput='uniform_average')
            val_loss += loss.item()
            val_r2 += r2.item()

    print(f"Epoch {epoch} | Train Loss {total_loss / len(train_loader)} | Val Loss {val_loss / len(val_loader)} | Train R2 {total_r2 / len(train_loader)} | Val R2 {val_r2 / len(val_loader)}")
    # Log training and validation loss to Weights and Biases
    wandb.log({"train_loss": total_loss / len(train_loader), 
               "val_loss": val_loss / len(val_loader), 
               "train_r2": total_r2 / len(train_loader),
               "val_r2": val_r2 / len(val_loader),
               "epoch": epoch})

# Save the trained model
torch.save(model.state_dict(), "rnn_model.pth")
# wandb.save("rnn_model.pth")  # Save the model to Weights and Biases

wandb.finish()

#%% ----------------------------------------------------------------------------
# Evaluate the trained model
# ------------------------------------------------------------------------------
outputs, gt_behav = [], []
from torcheval.metrics.functional import r2_score

model.eval()
for spikes, behavior in zip(val_spikes_tensor, val_behavior_tensor):
    val_loss, val_r2 = 0, 0
    with torch.no_grad():
        spikes, behavior = spikes.to(device), behavior.to(device)
        output = model(spikes)
        outputs.append(output)
        gt_behav.append(behavior)

        loss = mse_loss(output, behavior)
        val_loss += loss.item()

        r2_val = r2_score(output, behavior, multioutput='uniform_average')
        val_r2 += r2_val.item()


#%% ----------------------------------------------------------------------------
# Plot some trials
# ------------------------------------------------------------------------------
trials = [0, 1, 2, 3, 4]
fig, axs = plt.subplots(2, 5, figsize=(40, 10))
for idx, i in enumerate(range(5)):
    axs[0, idx].plot(outputs[i].cpu().numpy()[:, 0], label='pred')
    axs[0, idx].plot(gt_behav[i].cpu().numpy()[:, 0], label='gt')
    axs[0, idx].legend()
    axs[0, idx].set_title(f'X Velocity - Trial:{i}')
    axs[1, idx].plot(outputs[i].cpu().numpy()[:, 1], label='pred')
    axs[1, idx].plot(gt_behav[i].cpu().numpy()[:, 1], label='gt')
    axs[1, idx].set_title(f'Y Velocity - Trial:{i}')
plt.show()


# %%
