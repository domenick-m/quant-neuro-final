#%%
import os
import wandb
import numpy as np
import scipy.signal as signal
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nlb_tools.evaluation import evaluate
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import train_test_split
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5

#%%
# how much to delay the kinematics by
LAG = 40 # ms
lag_bins = int(LAG / 5) # number of bins to delay kinematics by

#%%
dataset_name = 'mc_maze' # NLB name of dataset
datapath = './data/000128/sub-Jenkins/' # Path to dataset

# download dataset if it doesn't exist
if not os.path.exists(datapath):
    print('Data not downloaded! Downloading now.')
    !dandi download https://dandiarchive.org/dandiset/000128 -o data


# %%
# function for gaussian smoothing
def smooth_data(data, kern_sd_ms=40):
    # data: (n_trials, n_timepoints, n_neurons)
    kern_sd = int(round(kern_sd_ms / dataset.bin_width))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    return np.apply_along_axis(filt, 1, data)


# %%
# load dataset and resample -- takes ~1.5 mins
dataset = NWBDataset(datapath)
dataset.resample(5) # resample to 5 ms bins


#%%
# create data dicts using NLB tools
# we are using the validation set as our test set
trainval_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split='train', save_file=False)
test_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split='val', save_file=False)
behav_dict = make_eval_target_tensors(dataset, dataset_name=dataset_name, train_trial_split='train', eval_trial_split='val', include_psth=False, save_file=False)


#%%
# create training set
trainval_spikes = np.concatenate((trainval_dict['train_spikes_heldin'], trainval_dict['train_spikes_heldout']), axis=2)
trainval_behavior = behav_dict['mc_maze']['train_behavior']

# smooth spikes
trainval_smth_spikes = smooth_data(trainval_spikes, 40) # smooth with 40ms std gauss

#%%
wandb.init(project='Quant Neuro Final Project', entity='domenick-m')

model = Ridge(alpha=1.0)

kf = KFold(n_splits = 5, shuffle = True, random_state = 1)

split_train_losses = []
split_val_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(trainval_smth_spikes)):
    X_train, X_val = trainval_smth_spikes[train_index], trainval_smth_spikes[val_index]
    y_train, y_val = trainval_behavior[train_index], trainval_behavior[val_index]

    # delay kinematics by LAG ms
    X_train, y_train = X_train[:, :-lag_bins], y_train[:, lag_bins:]

    X_train = X_train.reshape(-1, X_train.shape[2])
    X_val = X_val.reshape(-1, X_val.shape[2])
    y_train = y_train.reshape(-1, y_train.shape[2])
    y_val = y_val.reshape(-1, y_val.shape[2])

    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_loss = r2_score(y_train, train_preds)
    val_loss = r2_score(y_val, val_preds)

    split_train_losses.append(train_loss)
    split_val_losses.append(val_loss)
    
    # Logging losses to wandb
    wandb.log({f'split_{fold + 1}_train_loss': train_loss,
               f'split_{fold + 1}_val_loss': val_loss})
    
average_train_loss = np.mean(split_train_losses)
average_val_loss = np.mean(split_val_losses)

wandb.log({'average_train_loss': average_train_loss,
           'average_val_loss': average_val_loss})

X_trainval = trainval_smth_spikes.reshape(-1, trainval_smth_spikes.shape[2])
y_trainval = trainval_behavior.reshape(-1, trainval_behavior.shape[2])

model.fit(X_trainval, y_trainval)

trainval_preds = model.predict(X_trainval)

trainval_loss = r2_score(y_trainval, trainval_preds)

wandb.log({'trainval_loss': trainval_loss})

wandb.finish()

#%%
# create test set
test_spikes = np.concatenate((test_dict['eval_spikes_heldin'], test_dict['eval_spikes_heldout']), axis=2)
test_behavior = behav_dict['mc_maze']['eval_behavior']

print(test_spikes.shape)
print(test_behavior.shape)
