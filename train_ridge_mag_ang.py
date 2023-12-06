#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import yaml
import wandb
import numpy as np
from joblib import dump
from utils import prepare_datasets
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


#%% ----------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
# Load config parameters from file
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

PRED_THETA = True

# Log all runs to this wandb project
wand_proj_name = f'{config["WANDB_BASE_PROJECT"]}_ridge_mag_ang_theta'

# Prepare datasets (extract, smooth, lag, etc...)
trainval, test = prepare_datasets(config)
trainval_spikes, trainval_behavior = trainval

# Convert behavior to magnitude and angle
x_vel, y_vel = np.split(trainval_behavior, 2, axis=-1)
magnitude = np.sqrt(x_vel ** 2 + y_vel ** 2)
angle_rads = np.arctan2(y_vel, x_vel)
angle_sin, angle_cos = np.sin(angle_rads), np.cos(angle_rads)
trainval_behavior = np.concatenate([magnitude, angle_rads] if PRED_THETA else \
                                   [magnitude, angle_sin, angle_cos], axis=-1)


# ------------------------------------------------------------------------------
# Train Ridge Regression Models with K-Fold Cross-Validation
# ------------------------------------------------------------------------------
# Define lists to store best performing model
mag_models, ang_models, mag_model_losses, ang_model_losses = [], [], [], []

# Define the number of folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

for alpha in config["RIDGE_ALPHAS"]:
    # Initialize lists to store loss values for each fold
    train_mag_losses, train_ang_losses, train_r2 = [], [], []
    val_mag_losses, val_ang_losses, val_r2 = [], [], []

    for train_index, val_index in kf.split(trainval_spikes):
        # Split data into training and validation for the current fold
        train_spikes, val_spikes = trainval_spikes[train_index], trainval_spikes[val_index]
        train_behavior, val_behavior = trainval_behavior[train_index], trainval_behavior[val_index]

        # Flatten data (trials, time, neurons) -> (trials * time, neurons)
        train_spikes = train_spikes.reshape(-1, train_spikes.shape[-1])
        val_spikes = val_spikes.reshape(-1, val_spikes.shape[-1])
        train_behavior = train_behavior.reshape(-1, train_behavior.shape[-1])
        val_behavior = val_behavior.reshape(-1, val_behavior.shape[-1])

        # Train the model
        mag_model = Ridge(alpha=alpha)
        mag_model.fit(train_spikes, train_behavior[:, 0:1])
        ang_model = Ridge(alpha=alpha)
        ang_model.fit(train_spikes, train_behavior[:, 1:])

        # Get model outputs
        train_mag = mag_model.predict(train_spikes)
        train_ang = ang_model.predict(train_spikes)
        val_mag = mag_model.predict(val_spikes)
        val_ang = ang_model.predict(val_spikes)

        if not PRED_THETA:
            train_sin, train_cos = np.split(train_ang, 2, axis=-1)
            val_sin, val_cos = np.split(val_ang, 2, axis=-1)

        # magnitude loss
        train_mag_loss = np.mean((train_mag - train_behavior[:, 0]) ** 2)
        val_mag_loss = np.mean((val_mag - val_behavior[:, 0]) ** 2)
        train_mag_losses.append(train_mag_loss)
        val_mag_losses.append(val_mag_loss)

        # angle loss
        if PRED_THETA:
            train_sin_loss = np.mean((np.sin(train_ang) - np.sin(train_behavior[:, 1])) ** 2)
            train_cos_loss = np.mean((np.cos(train_ang) - np.cos(train_behavior[:, 1])) ** 2)
            val_sin_loss = np.mean((np.sin(val_ang) - np.sin(val_behavior[:, 1])) ** 2)
            val_cos_loss = np.mean((np.cos(val_ang) - np.cos(val_behavior[:, 1])) ** 2)
            train_ang_losses.append(train_sin_loss + train_cos_loss)
            val_ang_losses.append(val_sin_loss + val_cos_loss)
        else:
            train_sin_loss = np.mean((train_sin - np.sin(train_behavior[:, 1])) ** 2)
            train_cos_loss = np.mean((train_cos - np.cos(train_behavior[:, 2])) ** 2)
            val_sin_loss = np.mean((val_sin - np.sin(val_behavior[:, 1])) ** 2)
            val_cos_loss = np.mean((val_cos - np.cos(val_behavior[:, 2])) ** 2)
            train_ang_losses.append(train_sin_loss + train_cos_loss)
            val_ang_losses.append(val_sin_loss + val_cos_loss)

        # R^2 Loss
        if PRED_THETA:
            pred_train_x_vel = train_mag * np.cos(train_ang)
            pred_train_y_vel = train_mag * np.sin(train_ang)
            pred_val_x_vel = val_mag * np.cos(val_ang)
            pred_val_y_vel = val_mag * np.sin(val_ang)
        else:
            pred_train_x_vel = train_mag * train_cos
            pred_train_y_vel = train_mag * train_sin
            pred_val_x_vel = val_mag * val_cos
            pred_val_y_vel = val_mag * val_sin
        train_x_vel = train_behavior[:, 0] * np.cos(train_behavior[:, 1])
        train_y_vel = train_behavior[:, 0] * np.sin(train_behavior[:, 1])
        val_x_vel = val_behavior[:, 0] * np.cos(val_behavior[:, 1])
        val_y_vel = val_behavior[:, 0] * np.sin(val_behavior[:, 1])
        pred_train_vel = np.concatenate([pred_train_x_vel, pred_train_y_vel], axis=-1)
        pred_val_vel = np.concatenate([pred_val_x_vel, pred_val_y_vel], axis=-1)
        train_vel = np.concatenate([np.expand_dims(train_x_vel, 1), 
                                    np.expand_dims(train_y_vel, 1)], axis=1)
        val_vel = np.concatenate([np.expand_dims(val_x_vel, 1), 
                                  np.expand_dims(val_y_vel, 1)], axis=1)
        train_r2.append(r2_score(train_vel, pred_train_vel))
        val_r2.append(r2_score(val_vel, pred_val_vel))

    # Calculate average losses
    avg_mag_train_loss = np.mean(train_mag_losses)
    avg_ang_train_loss = np.mean(train_ang_losses)
    avg_mag_val_loss = np.mean(val_mag_losses)
    avg_ang_val_loss = np.mean(val_ang_losses)
    avg_train_r2 = np.mean(train_r2)
    avg_val_r2 = np.mean(val_r2)

    # Log model to WandB
    wandb.init(project=wand_proj_name, entity=config["WANDB_ENTITY"], name=f'alpha_{alpha}')
    wandb.log({'avg_mag_train_loss': avg_mag_train_loss, 
               'avg_ang_train_loss': avg_ang_train_loss,
               'avg_mag_val_loss': avg_mag_val_loss,
               'avg_ang_val_loss': avg_ang_val_loss,
               'avg_train_r2': avg_train_r2,
               'avg_val_r2': avg_val_r2})
    wandb.finish()

    # Store the model and val loss for this alpha
    mag_models.append(mag_model)
    ang_models.append(ang_model)
    mag_model_losses.append(avg_mag_val_loss)
    ang_model_losses.append(avg_ang_val_loss)

# Save the model to disk with the best validation loss
best_mag_model = mag_models[np.argmin(mag_model_losses)]
best_ang_model = ang_models[np.argmin(ang_model_losses)]
dump(best_mag_model, f'ridge_model_mag_{PRED_THETA}.joblib')
dump(best_ang_model, f'ridge_model_ang_{PRED_THETA}.joblib')
