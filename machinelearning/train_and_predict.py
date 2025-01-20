
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """
    Trains the neural network model.

    Parameters:
    model (nn.Module): The neural network model to be trained.
    train_loader (DataLoader): DataLoader for the training dataset.
    criterion (nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    epochs (int): Number of epochs to train the model.

    Returns:
    list: List of average loss values for each epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize a list to store the loss for each epoch
    loss_arr = []

    # Loop over the number of epochs
    for epoch in range(epochs):
        # Initialize the cumulative loss for the epoch
        epoch_loss = 0
        
        # Loop over batches in the DataLoader
        for X_batch, Y_batch in train_loader:
            # Zero the parameter gradients to prevent accumulation
            optimizer.zero_grad()
            
            # Perform the forward pass to get predictions
            y_hat = model(X_batch)
            
            # Compute the loss between predictions and true labels
            loss = criterion(y_hat, Y_batch)
            
            # Perform the backward pass to compute gradients
            loss.backward()
            
            # Update model parameters using the optimizer
            optimizer.step()
            
            # Accumulate the batch loss to the epoch loss
            epoch_loss += loss.item()
        
        # Compute the average loss for the epoch and add it to the loss array
        loss_arr.append(epoch_loss / len(train_loader))
        
        # Print the loss every 10 epochs for monitoring
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} Loss: {epoch_loss / len(train_loader):.4f}')
    
    return loss_arr


def predict_with_uncertainty(model, test_loader, n_iter=100):
    """
    Predicts with uncertainty estimation using Monte Carlo Dropout.

    Parameters:
    model (nn.Module): The trained neural network model.
    test_loader (DataLoader): DataLoader for the test dataset.
    n_iter (int): Number of iterations to perform Monte Carlo sampling.

    Returns:
    tuple: A tuple containing the mean and standard deviation of predictions.
    """
    # Set the model to training mode to enable dropout during inference
    model.train()

    # Initialize a list to store predictions from multiple forward passes
    predictions = []

    # Perform multiple forward passes
    for _ in range(n_iter):
        # Initialize a list to store predictions for the current iteration
        epoch_preds = []
        
        # Loop over batches in the DataLoader
        for X_batch, _ in test_loader:
            # Perform the forward pass and get predictions for the batch
            y_hat = model(X_batch).detach().cpu().numpy()
            
            # Append the predictions to the current iteration list
            epoch_preds.append(y_hat)
        
        # Stack the batch predictions and append to the main predictions list
        predictions.append(np.vstack(epoch_preds))
    
    # Convert the list of predictions to a NumPy array
    predictions = np.array(predictions)

    # Compute the mean of the predictions across the iterations
    prediction_mean = predictions.mean(axis=0)

    # Compute the standard deviation of the predictions across the iterations
    prediction_std = predictions.std(axis=0)

    return prediction_mean, prediction_std

