

import pandas as pd

import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Models.FCNNModel import FCNN
from sklearn.metrics import roc_auc_score
import numpy as np

trainX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_trainX.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_testX.pt')

trainY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN//ProcessedData/FCNN_trainY.csv')
testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_testY.csv')

# 1 = gender
# 2 = age
trainY = torch.tensor(trainY.iloc[:, 1].values)
testY = torch.tensor(testY.iloc[:, 1].values)

print(type(testY))


print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)




# Hyperparameters
input_dim = trainX.shape[1]
hidden_dim1 = 256
hidden_dim2 = 128
num_classes = len(torch.unique(trainY))
batch_size = 64
learning_rate = 1e-3
num_epochs = 10
dropout = 0.2
classification = True
decay = 0.001


train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

# Setting Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = FCNN(input_dim, hidden_dim1, hidden_dim2, num_classes, dropout, classification)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()


print("Training")
model.train_model(train_loader,criterion,optimizer,num_epochs)


print("Prediction")
eval = model.evaluate_model(test_loader)


all_preds,all_labels= model.predict_outputs(test_loader)


auc = roc_auc_score(all_labels, all_preds)

print(f'AUC: {auc}')

print(f'ACC: {eval}')
