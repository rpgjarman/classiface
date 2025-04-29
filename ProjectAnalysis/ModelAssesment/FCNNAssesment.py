

import pandas as pd

import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Models.FCNNModel import FCNN
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

trainX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_trainX.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_testX.pt')

trainY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN//ProcessedData/FCNN_trainY.csv')
testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/ProcessedData/FCNN_testY.csv')


# Hyperparameters
input_dim = trainX.shape[1]
hidden_dim1 = 256
hidden_dim2 = 128
num_classes = 0
batch_size = 64
learning_rate = 0.001
num_epochs = 10
dropout = 0.1
decay = 0.0001
classification = True
criterion = nn.CrossEntropyLoss()

test = input("Test: gender/age: ")

if test == "gender":
    trainY = torch.tensor(trainY.iloc[:, 1].values)
    testY = torch.tensor(testY.iloc[:, 1].values)
    classification = True
    criterion = nn.CrossEntropyLoss()
    num_classes = len(torch.unique(trainY))

elif test == "age":
    trainY = torch.tensor(trainY.iloc[:, 2].values)
    testY = torch.tensor(testY.iloc[:, 2].values)
    classification = False
    criterion = nn.MSELoss()
    num_classes = len(torch.unique(trainY))

else:
    print("Not a valid choice")



params = {'lr': 0.001,
 'h1': 256,
 'h2': 128,
 'batch_size': 64,
 'num_epochs': 10,
 'decay': 0.0001,
 'dropout': 0.1,
 'complexity': 256}

print(params)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)




train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)

# Setting Data Loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = FCNN(input_dim, hidden_dim1, hidden_dim2, num_classes, dropout, classification)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)



print("Training")
model.train_model(train_loader,criterion,optimizer,num_epochs)


print("Prediction")


if test == "gender":
    roc = {"fpr": [], "tpr": []}
    eval = model.evaluate_model(test_loader)
    all_preds, all_labels = model.predict_outputs(test_loader)
    yHat = (np.array(all_preds) >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    roc['fpr'], roc['tpr'], _ = roc_curve(all_labels, all_preds)

    print(f'AUC: {auc}')
    print(f"ROC Values: {roc}")

    txt = (f'/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/FCNN/FCNN_roc.csv')

    roc_df = pd.DataFrame(roc, columns=['fpr', 'tpr'])
    roc_df.to_csv(txt, index=False)

    print(f'ACC: {eval}')

if test == "age":
    eval = model.evaluate_model(test_loader)
    print(f'MSE: {eval}')









