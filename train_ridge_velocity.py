#%% ----------------------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------------------------
import yaml
import wandb
import numpy as np
from joblib import dump
from utils import prepare_datasets
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
    train_losses = []
    val_losses = []

    for train_index, val_index in kf.split(trainval_spikes):
        # Split data into training and validation for the current fold
        X_train, X_val = trainval_spikes[train_index], trainval_spikes[val_index]
        y_train, y_val = trainval_behavior[train_index], trainval_behavior[val_index]

        # Flatten data (trials, time, neurons) -> (trials * time, neurons)
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_val = X_val.reshape(-1, X_val.shape[-1])
        y_train = y_train.reshape(-1, y_train.shape[-1])
        y_val = y_val.reshape(-1, y_val.shape[-1])

        # Train the model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Calculate R^2 loss and store it
        train_loss = model.score(X_train, y_train)
        val_loss = model.score(X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Calculate average losses
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    # Log model to WandB
    wandb.init(project=wand_proj_name, entity=config["WANDB_ENTITY"], name=f'alpha_{alpha}')
    wandb.log({'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss})
    wandb.finish()

    # Store the model and val loss for this alpha
    models.append(model)
    model_losses.append(avg_val_loss)

# Save the model to disk with the best validation loss
best_model = models[np.argmin(model_losses)]
dump(best_model, 'ridge_model_velocity.joblib')
