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

PRED_THETA = False

# Log all runs to this wandb project
wand_proj_name = f'{config["WANDB_BASE_PROJECT"]}_ridge_mag_ang_no_theta'

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
models, model_losses = [], []

# Define the number of folds for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

for alpha in config["RIDGE_ALPHAS"]:
    # Initialize lists to store loss values for each fold
    train_losses, train_r2 = [], []
    val_losses, val_r2 = [], []

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
        model = Ridge(alpha=alpha)
        model.fit(train_spikes, train_behavior)

        # Get model outputs and split into mag and angle
        train_preds = model.predict(train_spikes)
        val_preds = model.predict(val_spikes)
        if PRED_THETA:
            train_mag, train_ang = np.split(train_preds, 2, axis=-1)
            val_mag, val_ang = np.split(val_preds, 2, axis=-1)
        else:
            train_mag, train_sin, train_cos = np.split(train_preds, 3, axis=-1)
            val_mag, val_sin, val_cos = np.split(val_preds, 3, axis=-1)
            train_ang = np.arctan2(train_sin, train_cos)
            val_ang = np.arctan2(val_sin, val_cos)

        # magnitude loss
        train_mag_loss = np.mean((train_mag - train_behavior[:, 0]) ** 2)
        val_mag_loss = np.mean((val_mag - val_behavior[:, 0]) ** 2)

        # angle loss
        if PRED_THETA:
            train_sin_loss = np.mean((np.sin(train_ang) - np.sin(train_behavior[:, 1])) ** 2)
            train_cos_loss = np.mean((np.cos(train_ang) - np.cos(train_behavior[:, 1])) ** 2)
            val_sin_loss = np.mean((np.sin(val_ang) - np.sin(val_behavior[:, 1])) ** 2)
            val_cos_loss = np.mean((np.cos(val_ang) - np.cos(val_behavior[:, 1])) ** 2)
            train_losses.append(train_mag_loss + train_sin_loss + train_cos_loss)
            val_losses.append(val_mag_loss + val_sin_loss + val_cos_loss)
        else:
            train_sin_loss = np.mean((train_sin - np.sin(train_behavior[:, 1])) ** 2)
            train_cos_loss = np.mean((train_cos - np.cos(train_behavior[:, 2])) ** 2)
            val_sin_loss = np.mean((val_sin - np.sin(val_behavior[:, 1])) ** 2)
            val_cos_loss = np.mean((val_cos - np.cos(val_behavior[:, 2])) ** 2)
            train_losses.append(train_mag_loss + train_sin_loss + train_cos_loss)
            val_losses.append(val_mag_loss + val_sin_loss + val_cos_loss)

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
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    avg_train_r2 = np.mean(train_r2)
    avg_val_r2 = np.mean(val_r2)

    # Log model to WandB
    wandb.init(project=wand_proj_name, entity=config["WANDB_ENTITY"], name=f'alpha_{alpha}')
    wandb.log({'avg_train_loss': avg_train_loss, 
               'avg_val_loss': avg_val_loss,
               'avg_train_r2': avg_train_r2,
               'avg_val_r2': avg_val_r2})
    wandb.finish()

    # Store the model and val loss for this alpha
    models.append(model)
    model_losses.append(avg_val_loss)

# Save the model to disk with the best validation loss
best_model = models[np.argmin(model_losses)]
dump(best_model, 'ridge_model_mag_angle.joblib')
