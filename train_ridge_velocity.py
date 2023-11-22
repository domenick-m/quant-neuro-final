#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import yaml
import wandb
import numpy as np
from joblib import dump
from utils import prepare_datasets
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


#%% ----------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
# Load config parameters from file
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Log all runs to this wandb project
wand_proj_name = f'{config["WANDB_BASE_PROJECT"]}_ridge_velocity'

# Prepare datasets (extract, smooth, lag, etc...)
trainval, test = prepare_datasets(config)
trainval_spikes, trainval_behavior = trainval


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

        # Calculate loss and r2 score and store them
        train_preds = model.predict(train_spikes)
        val_preds = model.predict(val_spikes)
        train_losses.append(mean_squared_error(train_behavior, train_preds))
        val_losses.append(mean_squared_error(val_behavior, val_preds))
        train_r2.append(r2_score(train_behavior, train_preds))
        val_r2.append(r2_score(val_behavior, val_preds))

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
dump(best_model, 'ridge_model_velocity.joblib')
