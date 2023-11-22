
import os
import numpy as np
from pathlib import Path
import scipy.signal as signal
from nlb_tools.nwb_interface import NWBDataset
from sklearn.model_selection import train_test_split
from nlb_tools.make_tensors import (
    make_train_input_tensors, 
    make_eval_input_tensors, 
    make_eval_target_tensors, 
)


def prepare_datasets(config):
    DATASET_IDS = {
        'mc_maze': 128,
        'mc_rtt': 129,
        'mc_maze_large': 138,
        'mc_maze_medium': 139,
        'mc_maze_small': 140,
    }
    # get the path to the NLB data using the dataset ID
    id = DATASET_IDS[config["DATASET"]]
    data_folder = Path.cwd() / 'data' / f'000{id}'

    # download dataset if it doesn't exist
    if not data_folder.exists():
        print(f'Dataset "{config["DATASET"]}" not downloaded! Downloading now.')
        os.makedirs(data_folder)
        os.system(f'dandi download https://dandiarchive.org/dandiset/000{id} -o data')

    # get the path to the folder inside the data folder
    data_path = next((p for p in data_folder.iterdir() if p.is_dir()), None)

    dataset = NWBDataset(data_path) # load dataset from file
    dataset.resample(config["BINSIZE"]) # resample to 5 ms bins

    # extract training, validation and test spikes
    trainval_dict = make_train_input_tensors(dataset, config["DATASET"], save_file=False) 
    test_dict = make_eval_input_tensors(dataset, config["DATASET"], save_file=False) 
    behav_dict = make_eval_target_tensors(dataset, config["DATASET"], save_file=False)

    # combine heldin and heldout neurons for trainval and test
    stacked = [trainval_dict[f'train_spikes_{i}'] for i in ['heldin', 'heldout']]
    trainval_spikes = np.concatenate(stacked, axis=2)
    stacked = [test_dict[f'eval_spikes_{i}'] for i in ['heldin', 'heldout']]
    test_spikes = np.concatenate(stacked, axis=2)

    # smooth spikes using gaussian window
    kern_sd = int(round(config["SMOOTH_SD"] / dataset.bin_width))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    filt = lambda x: np.convolve(x, window / np.sum(window), 'same')
    trainval_smth_spikes = np.apply_along_axis(filt, 1, trainval_spikes)
    test_smth_spikes = np.apply_along_axis(filt, 1, test_spikes)

    # extract training, validation, and test kinematics
    trainval_behavior = behav_dict[config["DATASET"]]['train_behavior']
    test_behavior = behav_dict[config["DATASET"]]['eval_behavior']

    # delay kinematics by LAG ms
    lag_bins = int(config["LAG"] / config["BINSIZE"]) # number of bins to delay kinematics by
    trainval_smth_spikes = trainval_smth_spikes[:, :-lag_bins]
    trainval_behavior = trainval_behavior[:, lag_bins:]
    test_smth_spikes = test_smth_spikes[:, :-lag_bins]
    test_behavior = test_behavior[:, lag_bins:]

    return (trainval_smth_spikes, trainval_behavior), (test_smth_spikes, test_behavior)

