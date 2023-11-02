#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import os
import wandb
import numpy as np
from pathlib import Path
import scipy.signal as signal
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nlb_tools.evaluation import evaluate
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import train_test_split
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5


#%% ----------------------------------------------------------------------------
# Static Parameters
# ------------------------------------------------------------------------------
BINSIZE = 5 # ms - bin size for resampling
LAG = 100 # ms - how much to delay the kinematics by
SMOOTH_SD = 40 # ms - std of gaussian kernel for smoothing
DATASET = 'mc_maze_small' # NLB name of dataset

WANDB_PROJECT = '7610 Final Project Testing' # name of wandb project
WANDB_ENTITY = 'domenick-m' # wandb username (change to yours)


#%% ----------------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------------------
RIDGE_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1, 10] # alphas for ridge regression

RNN_LAYERS = [1, 2, 3, 4, 5] # number of layers to try for RNN
RNN_HIDDEN_SIZE = [32, 64, 128, 256] # hidden size for RNN
RNN_DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # dropout for RNN
RNN_LR = [0.0001, 0.001, 0.01] # learning rate for RNN
RNN_EPOCHS = [100, 500, 1000] # number of epochs to train RNN for
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
# get the path to the NLB data using the dataset ID
data_folder = Path.cwd() / 'data' / f'000{DATASET_IDS[DATASET]}'

# download dataset if it doesn't exist
if not data_folder.exists():
    print(f'Dataset "{DATASET}" not downloaded! Downloading now.')
    !dandi download f'https://dandiarchive.org/dandiset/000{id} -o data'

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

# # smooth spikes using gaussian window
# kern_sd = int(round(SMOOTH_SD / dataset.bin_width))
# window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
# filt = lambda x: np.convolve(x, window / np.sum(window), 'same')
# trainval_smth_spikes = np.apply_along_axis(filt, 1, trainval_spikes)

# # extract training and validation kinematics
# trainval_behavior = behav_dict[DATASET]['train_behavior']

# # delay kinematics by LAG ms
# lag_bins = int(LAG / BINSIZE) # number of bins to delay kinematics by
# trainval_smth_spikes = trainval_smth_spikes[:, :-lag_bins]
# trainval_behavior = trainval_behavior[:, lag_bins:]

# # split into training and validation sets
# tv_splits = train_test_split(trainval_smth_spikes, 
#                              trainval_behavior, 
#                              test_size=0.2, 
#                              random_state=1)
# train_spikes, val_spikes, train_behavior, val_behavior = tv_splits

# # flatten data (trials, time, neurons) -> (trials * time, neurons)
# train_spikes = train_spikes.reshape(-1, train_spikes.shape[-1])
# val_spikes = val_spikes.reshape(-1, val_spikes.shape[-1])
# train_behavior = train_behavior.reshape(-1, train_behavior.shape[-1])
# val_behavior = val_behavior.reshape(-1, val_behavior.shape[-1])

# #%% ----------------------------------------------------------------------------
# # Train Ridge Regression Models
# # ------------------------------------------------------------------------------
# for alpha in RIDGE_ALPHAS:
#     # start a new wandb run
#     wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f'lr_{alpha}')

#     # create and fit the ridge regression model
#     model = Ridge(alpha=alpha)
#     model.fit(train_spikes, train_behavior)

#     # score the model models using R^2 loss and log to wandb
#     train_loss = model.score(train_spikes, train_behavior)
#     val_loss = model.score(val_spikes, val_behavior)
#     wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

#     # finish the wandb run before starting a new one
#     wandb.finish()

# # ------------------------------------------------------------------------------
# # Train Ridge Regression Models with K-Fold Cross-Validation
# # ------------------------------------------------------------------------------
# from sklearn.metrics import mean_squared_error

# # Define the number of folds for cross-validation
# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# for alpha in RIDGE_ALPHAS:
#     # Initialize lists to store loss values for each fold
#     train_losses = []
#     val_losses = []

#     for train_index, val_index in kf.split(trainval_smth_spikes):
#         # Split data into training and validation for the current fold
#         X_train, X_val = trainval_smth_spikes[train_index], trainval_smth_spikes[val_index]
#         y_train, y_val = trainval_behavior[train_index], trainval_behavior[val_index]

#         # Flatten data
#         X_train = X_train.reshape(-1, X_train.shape[-1])
#         X_val = X_val.reshape(-1, X_val.shape[-1])
#         y_train = y_train.reshape(-1, y_train.shape[-1])
#         y_val = y_val.reshape(-1, y_val.shape[-1])

#         # Train the model
#         model = Ridge(alpha=alpha)
#         model.fit(X_train, y_train)

#         # Calculate R^2 loss and store it
#         train_loss = model.score(X_train, y_train)
#         val_loss = model.score(X_val, y_val)
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#     # Calculate average losses
#     avg_train_loss = np.mean(train_losses)
#     avg_val_loss = np.mean(val_losses)

#     # Log to WandB
#     wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f'ridge_alpha_{alpha}')
#     wandb.log({'average_train_loss': avg_train_loss, 'average_val_loss': avg_val_loss})
#     wandb.finish()

#%%
# New parameters for sweeping
LAG_VALUES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]  # Different LAG values to try
SMOOTH_SD_VALUES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # Different SMOOTH_SD values to try

# KFold setup
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS)

# Main training loop
for lag in LAG_VALUES:
    for smooth_sd in SMOOTH_SD_VALUES:
        # Adjust this part to process data with current lag and smooth_sd
        lag_bins = int(lag / BINSIZE)
        kern_sd = int(round(smooth_sd / dataset.bin_width))

        # smooth spikes using gaussian window
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        filt = lambda x: np.convolve(x, window / np.sum(window), 'same')
        trainval_smth_spikes = np.apply_along_axis(filt, 1, trainval_spikes)

        # extract training and validation kinematics
        trainval_behavior = behav_dict[DATASET]['train_behavior']

        # delay kinematics by LAG ms
        trainval_smth_spikes_lagged = trainval_smth_spikes[:, :-lag_bins]
        trainval_behavior_lagged = trainval_behavior[:, lag_bins:]

        for alpha in RIDGE_ALPHAS:
            fold_train_losses = []
            fold_val_losses = []

            for train_index, val_index in kf.split(trainval_smth_spikes_lagged):
                # Split data into kth fold training and validation sets
                X_train, X_val = trainval_smth_spikes_lagged[train_index], trainval_smth_spikes_lagged[val_index]
                y_train, y_val = trainval_behavior_lagged[train_index], trainval_behavior_lagged[val_index]

                # Flatten data
                X_train = X_train.reshape(-1, X_train.shape[-1])
                X_val = X_val.reshape(-1, X_val.shape[-1])
                y_train = y_train.reshape(-1, y_train.shape[-1])
                y_val = y_val.reshape(-1, y_val.shape[-1])

                # Train model
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)

                # Compute losses for this fold
                train_loss = model.score(X_train, y_train)
                val_loss = model.score(X_val, y_val)
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)

            # Average losses over all folds
            avg_train_loss = np.mean(fold_train_losses)
            avg_val_loss = np.mean(fold_val_losses)

            # Log to wandb
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f'ridge_alpha_{alpha}_lag_{lag}_smooth_{smooth_sd}')
            wandb.log({
                'alpha': alpha,
                'lag': lag,
                'smooth_sd': smooth_sd,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss
            })
            wandb.finish()
