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
DATASET = 'mc_maze_small' # NLB name of dataset

WANDB_PROJECT = 'quantNeuroTest' # name of wandb project
WANDB_ENTITY = 'forrealahmad' # wandb username (change to yours)
PATIENCE = 5
MIN_EPOCH = 100


#%% ----------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------

RNN_LAYERS = [1, 2, 3, 4, 5] # number of layers to try for RNN
RNN_HIDDEN_SIZE = [32, 64, 128, 256] # hidden size for RNN
RNN_DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # dropout for RNN
RNN_LR = [0.001, 0.01] # learning rate for RNN
RNN_EPOCHS = [1000] # number of epochs to train RNN for
RNN_BATCH_SIZE = [32, 64, 128, 256] # batch size for RNN

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
print(data_path)
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


#%% -----------------------------------
# Run a sweep on hyperparameters
# -------------------------------------

# Set up the sweep configuration
sweep_config = {
    "name": "rnn-random_sweep03",
    "method": "random",
    "parameters": {
        "num_layers": {"values": RNN_LAYERS},
        "hidden_size": {"values": RNN_HIDDEN_SIZE},
        "dropout": {"values": RNN_DROPOUT},
        "lr": {"values": RNN_LR},
        "rnn_epochs": {"values": RNN_EPOCHS},
        "batch_size": {"values": RNN_BATCH_SIZE}
    }
}

#%% -------------------------------------
# Custom Loss
# ---------------------------------------
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, predicted_vectors, target_vectors):
        # Compute cosine similarity between predicted and target vectors
        cos_sim = self.cosine_similarity(predicted_vectors, target_vectors)
        # Compute 1 minus cosine similarity as the loss
        loss = 1 - cos_sim.mean()
        return loss

#%%
best_val_loss = float('inf')

def train_sweep(config=None):


    best_val_loss = float('inf')

    with wandb.init(config=config):
        print(wandb.config)
        num_layers = wandb.config.num_layers
        hidden_size = wandb.config.hidden_size
        dropout = wandb.config.dropout
        lr = wandb.config.lr
        rnn_epochs = wandb.config.rnn_epochs
        batch_size = wandb.config.batch_size


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

        # Set up the RNN model with the current configuration
        model = RNNMODEL(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)

        mse_loss = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(rnn_epochs):
            model.train()
            total_loss = 0
            for batch_idx, (spikes, behavior) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(spikes)
                loss = mse_loss(output, behavior)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log the training loss to Weights and Biases
            wandb.log({"train_loss": total_loss / len(train_loader), "epoch": epoch})

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (spikes, behavior) in enumerate(val_loader):
                    output = model(spikes)
                    loss = mse_loss(output, behavior)
                    val_loss += loss.item()

            # Log the validation loss to Weights and Biases
            wandb.log({"ave_val_loss": val_loss / len(val_loader),  "epoch": epoch})
                # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= PATIENCE and epoch>=MIN_EPOCH:
                    print("Early stopping triggered.")
                    break
# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project='QunatNeuroRNNSweep')
# Run the sweep
wandb.agent(sweep_id, function=train_sweep)

