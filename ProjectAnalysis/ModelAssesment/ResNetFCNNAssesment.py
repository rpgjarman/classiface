from Models.ResNetFCNN import ResNetFCNN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import time


# trainX1 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX0.pt')
# trainX2 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX1.pt')
# trainX3 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX2.pt')
# trainX4 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX3.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testX.pt')

trainY1 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY0.csv')
trainY2 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY1.csv')
trainY3 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY2.csv')
trainY4 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY3.csv')

testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testY.csv')






hidden_dim1=128
hidden_dim2= 32
num_classes = 0
batch_size= 128
lr =0.001
num_epochs=10
dropout= 0.1
decay=1e-06
classification = True
criterion = nn.CrossEntropyLoss()


params = {"lr" : lr,
            "h1" : hidden_dim1,
            "h2" : hidden_dim2,
            "batch_size" : batch_size,
            "num_epochs" : num_epochs,
            'decay': decay,
            'dropout': dropout,}





test = input("Test: gender/age ")

if test == "gender":
    print("Test Gender")
    trainY1 = torch.tensor(trainY1.iloc[:, 1].values)
    trainY2 = torch.tensor(trainY2.iloc[:, 1].values)
    trainY3 = torch.tensor(trainY3.iloc[:, 1].values)
    trainY4 = torch.tensor(trainY4.iloc[:, 1].values)
    testY = torch.tensor(testY.iloc[:, 1].values)
    classification = True
    criterion = nn.CrossEntropyLoss()
elif test == "age":
    print("Test Age")
    trainY1 = torch.tensor(trainY1.iloc[:, 2].values)
    trainY2 = torch.tensor(trainY2.iloc[:, 2].values)
    trainY3 = torch.tensor(trainY3.iloc[:, 2].values)
    trainY4 = torch.tensor(trainY4.iloc[:, 2].values)
    testY = torch.tensor(testY.iloc[:, 2].values)
    classification = False
    criterion = nn.MSELoss()
else:
    print("Not a valid choice")

trainY_list = [trainY1, trainY2, trainY3, trainY4]

all_trainY = torch.cat(trainY_list)

num_classes = len(torch.unique(all_trainY))

print(f"Testing: {params}")


# Loading Data
model = ResNetFCNN(hidden_dim1, hidden_dim2, num_classes, dropout, classification=classification)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
for i in range(0,4):
    path = (f'/Users/damienlo/Desktop/University/CS 334/'
            f'Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX{i}.pt')
    trainX = torch.load(path)
    train_dataset = TensorDataset(trainX, trainY_list[i])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training on dataset {i+1}/4")
    model.train_model(train_loader, criterion, optimizer, num_epochs)

test_dataset = TensorDataset(testX, testY)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


if test == "gender":
    roc = {"fpr": [], "tpr": []}
    eval = model.evaluate_model(test_loader)
    all_preds, all_labels = model.predict_outputs(test_loader)
    yHat = (np.array(all_preds) >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    roc['fpr'], roc['tpr'], _ = roc_curve(all_labels, all_preds)

    print(f'AUC: {auc}')
    print(f"ROC Values: {roc}")

    txt = (f'/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ResNetFCNN_roc.csv')

    roc_df = pd.DataFrame(roc, columns=['fpr', 'tpr'])
    roc_df.to_csv(txt, index=False)

    print(f'ACC: {eval}')

if test == "age":
    eval = model.evaluate_model(test_loader)
    print(f'MSE: {eval}')