#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import os
import yaml
import wandb
import numpy as np
from joblib import load
from joblib import dump
from utils import prepare_datasets
from sklearn.metrics import r2_score, mean_squared_error
from rnn_model import RNNModel
import torch


#%% ----------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
# Load config parameters from file
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Log all runs to this wandb project
wand_proj_name = f'{config["WANDB_BASE_PROJECT"]}_test_eval'

# Prepare datasets (extract, smooth, lag, etc...)
trainval, test = prepare_datasets(config)
trainval_spikes, trainval_behavior = trainval
test_spikes, test_behavior = trainval
print(test_behavior.shape)


#%% ----------------------------------------------------------------------------
# Load the ridge models
# ------------------------------------------------------------------------------
ridge_model_velocity = load('ridge_model_velocity.joblib')
ridge_model_angle = load('ridge_model_ang_False.joblib')
ridge_model_mag = load('ridge_model_mag_False.joblib')


#%% ----------------------------------------------------------------------------
# Load the RNN models
# ------------------------------------------------------------------------------
# Find the mag ang runs with the lowest validation losses
api = wandb.Api()
runs = api.runs(f'{config["WANDB_ENTITY"]}/quant_neuro_final_rnn_mag_ang_no_theta')
lowest_mag_val_loss = float('inf')
lowest_ang_val_loss = float('inf')
best_mag_run_name = None
best_ang_run_name = None
best_mag_config = None
best_ang_config = None
for run in runs:
    mag_val_loss = run.summary.get('mag_val_loss', None)
    ang_val_loss = run.summary.get('ang_val_loss', None)
    if mag_val_loss is not None and mag_val_loss < lowest_mag_val_loss:
        lowest_mag_val_loss = mag_val_loss
        best_mag_run_name = run.name
        best_mag_config = run.config
    if ang_val_loss is not None and ang_val_loss < lowest_ang_val_loss:
        lowest_ang_val_loss = ang_val_loss
        best_ang_run_name = run.name
        best_ang_config = run.config
best_mag_run_name = 'serene-sweep-55'
best_ang_run_name = 'expert-sweep-24'
best_mag_model_path = os.path.join('./wandb_runs/rnn_mag_ang', f'{best_mag_run_name}-mag-best.pth')
best_ang_model_path = os.path.join('./wandb_runs/rnn_mag_ang', f'{best_ang_run_name}-ang-best.pth')
mag_model = RNNModel(input_size=test_spikes.shape[-1], 
                     hidden_size=best_mag_config['hidden_size'],
                     num_layers=best_mag_config['n_layers'],
                     dropout=best_mag_config['dropout'],
                     n_outputs=1)
ang_model = RNNModel(input_size=test_spikes.shape[-1],
                     hidden_size=best_ang_config['hidden_size'],
                     num_layers=best_ang_config['n_layers'],
                     dropout=best_ang_config['dropout'])
mag_model.load_state_dict(torch.load(best_mag_model_path))
ang_model.load_state_dict(torch.load(best_ang_model_path))
mag_model.eval()
ang_model.eval()
mag_model.double()
ang_model.double()

# Find the velocity run with the lowest validation loss
runs = api.runs(f'{config["WANDB_ENTITY"]}/quant_neuro_final_rnn_velocity')
lowest_val_loss = float('inf')
best_run_name = None
best_run_config = None
for run in runs:
    val_loss = run.summary.get('val_loss', None)
    if val_loss is not None and val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        best_run_name = run.name
        best_run_config = run.config
best_vel_model_path = os.path.join('./wandb_runs/rnn_velocity', f'{best_run_name}-best.pth')
vel_model = RNNModel(input_size=test_spikes.shape[-1],
                     hidden_size=best_run_config['hidden_size'],
                     num_layers=best_run_config['n_layers'],
                     dropout=best_run_config['dropout'])
vel_model.load_state_dict(torch.load(best_vel_model_path))
vel_model.eval()
vel_model.double()


#%% ----------------------------------------------------------------------------
# Evaluate the trained models on the test set
# ------------------------------------------------------------------------------

with torch.no_grad():
    # Ridge model
    spikes = test_spikes.reshape(-1, test_spikes.shape[-1])
    ridge_vel_out = ridge_model_velocity.predict(spikes)
    ridge_mag_out = ridge_model_mag.predict(spikes)
    ridge_ang_out = ridge_model_angle.predict(spikes)
    sin_out, cos_out = np.split(ridge_ang_out, 2, axis=-1)
    ridge_mag_ang_out = (np.concatenate([cos_out, sin_out], axis=1) * ridge_mag_out)

    ridge_vel_r2 = r2_score(test_behavior.reshape((-1, 2)), ridge_vel_out)
    ridge_mag_ang_r2 = r2_score(test_behavior.reshape((-1, 2)), ridge_mag_ang_out)
    ridge_vel_out = ridge_vel_out.reshape((-1, test_spikes.shape[1], 2))
    ridge_mag_ang_out = ridge_mag_ang_out.reshape((-1, test_spikes.shape[1], 2))

    # RNN model
    spikes = torch.tensor(test_spikes, dtype=torch.float64)
    rnn_vel_out = vel_model(spikes).numpy()
    rnn_mag_out = mag_model(spikes).numpy()
    rnn_ang_out = ang_model(spikes).numpy()
    sin_out, cos_out = np.split(rnn_ang_out, 2, axis=-1)
    rnn_mag_ang_out = np.concatenate([cos_out, sin_out], axis=2) * rnn_mag_out
    
    rnn_vel_r2 = r2_score(test_behavior.reshape((-1, 2)), rnn_vel_out.reshape((-1, 2)))
    rnn_mag_ang_r2 = r2_score(test_behavior.reshape((-1, 2)), rnn_mag_ang_out.reshape((-1, 2)))

    print(f'Ridge Velocity R2: {ridge_vel_r2}')
    print(f'Ridge Mag Ang R2: {ridge_mag_ang_r2}')

    print(f'RNN Velocity R2: {rnn_vel_r2}')
    print(f'RNN Mag Ang R2: {rnn_mag_ang_r2}')


#%% ----------------------------------------------------------------------------
# # Plot some trials
# # ------------------------------------------------------------------------------
import matplotlib.pyplot as plt
trials = [5, 10, 20]

for model, name in zip([ridge_vel_out, ridge_mag_ang_out, rnn_vel_out, rnn_mag_ang_out],
                       ['Ridge Velocity Model', 'Ridge Magnitude Angle Model', 
                        'RNN Velocity Model', 'RNN Magitude Angle Model']):
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    print(model.shape)
    for idx, i in enumerate(trials):
        axs[0, idx].plot(model[i, :, 0], label='pred')
        axs[0, idx].plot(test_behavior[i][:, 0], label='true')
        axs[0, idx].legend()
        axs[0, idx].set_title(f'X Velocity - Trial:{i}')
        axs[1, idx].plot(model[i, :, 1], label='pred')
        axs[1, idx].plot(test_behavior[i][:, 1], label='true')
        # set axis title for x axis
        axs[1, idx].set_title(f'Y Velocity - Trial:{i}')
        axs[1, idx].set_xlabel('Time in Trial (ms)')
    plt.suptitle(name+' Predictions vs True Kinematics')
    plt.tight_layout()
    # give space
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    plt.savefig(f'{name}.png', facecolor='white', transparent=False)
    plt.show()

# %%
