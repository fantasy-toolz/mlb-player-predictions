
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network with dropout
class ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANN, self).__init__()
        # First fully connected layer: 7 input features, 64 output features
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        
        # Batch normalization for the first layer
        self.bn1 = nn.BatchNorm1d(64)
        
        # Second fully connected layer: 64 input features, 32 output features (with batch normalisation)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)

        # Third fully connected layer: 32 input features, 16 output features (with batch normalisation)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.bn3 = nn.BatchNorm1d(16)
        
        # Output layer: 16 input features, 23 output features
        # Ensure the number of output neurons matches the problem. For classification tasks, the number of output neurons should match the number of classes.
        # here, 23 is the possible range of wins in the season
        self.output = nn.Linear(in_features=16, out_features=num_classes)
        self.num_classes = num_classes
        
        # Dropout layer with 50% dropout rate
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Apply first fully connected layer, batch normalization, ReLU activation, and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        # Apply second fully connected layer, batch normalization, ReLU activation, and dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        # Apply third fully connected layer, batch normalisation, ReLU activiation, and dropout
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # Apply output layer
        x = self.output(x)

        # normalise outputs
        if self.num_classes>1:
            return F.softmax(x, dim=1)
        else:
            return x
