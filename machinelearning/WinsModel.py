
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from ann_architecture import *
from train_and_predict import *

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# this will make the output exactly the same
set_seed(1)  # You can choose any integer as the seed

#import src.predictiondata as ss
from scipy import stats

year = 2022
#P = ss.grab_fangraphs_pitching_data([year])

P = pd.read_csv('data/{}pitching.csv'.format(year))

# try number of games and innings
# try number of games started (to indicate starters)
# try team winning percentage as a marker of quality


tbfnum = (P['TBF'].values)


tbfnum = (P['TBF'].values.astype('float'))
tbflimit=100

tbfnorm = (P['TBF'].values.astype('float'))[tbfnum>tbflimit]

gsnorm = (P['GS'].values.astype('float'))[tbfnum>tbflimit]
svnorm = (P['SV'].values.astype('float'))[tbfnum>tbflimit]
hrnum = (P['HR'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
innnum = (P['IP'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm

hitnum = (P['H'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
bbarr = (P['BB'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
eraarr = (P['ER'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
karr = (P['SO'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
team = P['Team'].values.astype('str')[tbfnum>tbflimit]
plrs = (P['Name'].values)[tbfnum>tbflimit]

# Wins are the target
warr = (P['W'].values.astype('float'))[tbfnum>tbflimit]#/tbfnorm


data = pd.DataFrame({
    'feature1': svnorm,
    'feature2': gsnorm,
    'feature3': eraarr,
    'feature4': hitnum,
    'feature5': karr,
    'feature6': bbarr,
    'feature7': hrnum,
    #'categorical_feature': team,
    'target': warr
})

# Separate features and target
X = data.drop('target', axis=1).values
Y = data['target'].values


"""
# Encode a categorical feature
label_encoder = LabelEncoder()
print(X['categorical_feature'])
X['categorical_feature_encoded'] = label_encoder.fit_transform(X['categorical_feature'])

# One-hot encoding the categorical feature
onehot_encoder = OneHotEncoder(sparse=False)
categorical_encoded = onehot_encoder.fit_transform(X[['categorical_feature_encoded']])

print(categorical_encoded)

# Drop original categorical columns and add the one-hot encoded columns
X = X.drop(['categorical_feature', 'categorical_feature_encoded'], axis=1)
X = np.concatenate((X.values, categorical_encoded),axis=1)
"""

print(X.shape,Y.shape)

# Create an array of indices
indices = np.arange(len(X))

# Split and standardize the data
X_train, X_test, Y_train, Y_test, train_indices, test_indices = train_test_split(X, Y, indices, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(f'Training indices: {train_indices}')
#print(f'Test indices: {test_indices}')

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Step 1: Normalize the target values (optional but recommended)
# Example: Min-Max scaling to scale the target values between 0 and 1
Y_min = Y.min()
Y_max = Y.max()
Y_train_normalized = (Y_train - Y_min) / (Y_max - Y_min)
Y_test_normalized = (Y_test - Y_min) / (Y_max - Y_min)

# Step 2: Convert to PyTorch tensors
Y_train = torch.tensor(Y_train_normalized, dtype=torch.float32).reshape(-1, 1)  # Reshape to ensure it's a column vector
Y_test = torch.tensor(Y_test_normalized, dtype=torch.float32).reshape(-1, 1)    # Reshape to ensure it's a column vector


#Y_train = torch.LongTensor(Y_train)
#Y_test = torch.LongTensor(Y_test)

Y_train = torch.FloatTensor(Y_train)
Y_test = torch.FloatTensor(Y_test)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize model, loss function, and optimizer
input_size = len(X_train[0])  # Assuming X_train is your training input data

#num_classes = 23#len(set(Y_train[0]))  # Assuming Y_train is your training target data
#model = ANN(input_size, num_classes)
#criterion = nn.CrossEntropyLoss()


num_classes = 1#len(set(Y_train[0]))  # Assuming Y_train is your training target data
model = ANN(input_size, num_classes)
criterion = nn.MSELoss()

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adding L2 regularization


def train_multiple_models(n_models, X_train, Y_train, X_test):
    models = []
    for _ in range(n_models):
        model = ANN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer)
        models.append(model)
    return models

def ensemble_predict(models, X_test):
    predictions = []
    for model in models:
        with torch.no_grad():
            preds = model(X_test).cpu().numpy()
            predictions.append(preds)
    return np.mean(predictions, axis=0)


# Accuracy calculation function
def calculate_accuracy(Y_true, Y_pred, tolerance=2):
    correct = (abs(Y_true - Y_pred) <= tolerance).sum()
    return correct / len(Y_true)

# Train multiple models and use ensemble for prediction
#models = train_multiple_models(5, X_train, Y_train, X_test)
#ensemble_predictions = ensemble_predict(models, X_test)


# Train the model
loss_arr = train_model(model, train_loader, criterion, optimizer)

# Save the model
torch.save(model.state_dict(), 'model_01Jun24.dat')

# Predict and calculate uncertainty
with torch.no_grad():
    pred_mean, pred_std = predict_with_uncertainty(model, test_loader)

# Convert predictions to class labels: pick the most probable value
preds = np.argmax(pred_mean, axis=1)

print(pred_mean)

pred_mean *= Y_max
pred_std *= Y_max

#print(pred_mean.shape,preds)
#print(Y_test.numpy())
#print(P['Name'].values[test_indices])

for indx in range(0,pred_mean.shape[0]):
    #print('{0:2d},{1:2d},{2:30s}'.format(Y_test.numpy()[indx],preds[indx],plrs[test_indices][indx]))
    print('{0:2d},{1:4.2f},{2:4.2f},{3:30s}'.format(int(Y_test.numpy()[indx][0]*Y_max),pred_mean[indx][0],pred_std[indx][0],plrs[test_indices][indx]))


print(pred_std.shape)

# Calculate and print accuracy
accuracy = calculate_accuracy(Y_test.numpy(), preds)
print(f'Accuracy: {accuracy:.4f}')


#now let's read the model in and see what we got

model.eval()


year = 2024
#P = ss.grab_fangraphs_pitching_data([year])

P = pd.read_csv('data/{}pitching.csv'.format(year))

tbfnum = (P['TBF'].values)

tbfnum = (P['TBF'].values.astype('float'))
tbflimit=100

tbfnorm = (P['TBF'].values.astype('float'))[tbfnum>tbflimit]

gsnorm = (P['GS'].values.astype('float'))[tbfnum>tbflimit]
svnorm = (P['SV'].values.astype('float'))[tbfnum>tbflimit] # add in BS here
hrnum = (P['HR'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
hitnum = (P['H'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
bbarr = (P['BB'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
eraarr = (P['ER'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
karr = (P['SO'].values.astype('float'))[tbfnum>tbflimit]/tbfnorm
team = P['Team'].values.astype('str')[tbfnum>tbflimit]
plrs = (P['Name'].values)[tbfnum>tbflimit]

# Wins are the target
warr = (P['W'].values.astype('float'))[tbfnum>tbflimit]#/tbfnorm


data = pd.DataFrame({
    'plrs':plrs,
    'feature1': svnorm,
    'feature2': gsnorm,
    'feature3': eraarr,
    'feature4': hitnum,
    'feature5': karr,
    'feature6': bbarr,
    'feature7': hrnum,
    #'categorical_feature': team,
    'target': warr
})

# Separate features and target
Xnew = data.drop(['plrs','target'], axis=1).values
Y = data['target'].values

new_data_normalized = sc.transform(Xnew)
new_data_tensor = torch.FloatTensor(new_data_normalized)

def predict_with_uncertainty(model, inputs, n_iter=100):
    model.train()  # Set the model to training mode to enable dropout
    predictions = []

    for _ in range(n_iter):
        with torch.no_grad():
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.array(predictions)
    mean_predictions = predictions.mean(axis=0)
    uncertainty = predictions.std(axis=0)

    return mean_predictions, uncertainty

# Get predictions and uncertainties
mean_predictions, uncertainties = predict_with_uncertainty(model, new_data_tensor)

#print(f'Predicted class: {predicted_class}')
for indx in range(0,pred_mean.shape[0]):
    print('{0:2d},{1:4.2f},{2:4.2f},{3:30s}'.format(int(Y[indx]),np.round(mean_predictions[indx][0]*Y_max,2),np.round(uncertainties[indx][0]*Y_max,2),data['plrs'].values[indx]))

