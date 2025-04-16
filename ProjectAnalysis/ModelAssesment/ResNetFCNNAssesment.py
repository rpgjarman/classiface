from Models.ResNetFCNN import ResNetFCNN
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,f1_score
import matplotlib.pyplot as plt
import numpy as np


param_tracker = '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/RES_HyperParamLog.csv'


trainX1 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX1.pt')
trainX2 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX2.pt')
trainX3 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX3.pt')
trainX4 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX4.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testX.pt')

trainY1 = torch.tensor(pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY1.csv').iloc[:, 1].values)
trainY2 = torch.tensor(pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY2.csv').iloc[:, 1].values)
trainY3 = torch.tensor(pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY3.csv').iloc[:, 1].values)
trainY4 = torch.tensor(pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY4.csv').iloc[:, 1].values)

testY = torch.tensor(pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testY.csv').iloc[:, 2].values)

trainX_list = [trainX1, trainX2, trainX3, trainX4]
trainY_list = [trainY1, trainY2, trainY3, trainY4]

all_trainY = torch.cat(trainY_list)



# Hyperparameters
lr = 1
hidden_dim1 = 1
hidden_dim2 = 1
batch_size = 1
dropout = 1
decay = 1
num_epochs = 1
num_classes = len(torch.unique(all_trainY))



train_loader_list = []
for i in range(len(trainX_list)):
    trainX = trainX_list[i]
    train_dataset = TensorDataset(trainX_list[i], trainY_list[i])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Training
model = ResNetFCNN(hidden_dim1, hidden_dim2, num_classes, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
criterion = nn.CrossEntropyLoss()

print(f'BEGINING TRANING')
model.train_model(train_loader_list, criterion, optimizer, num_epochs)
print('ENDING TRANING')


print('BEGINING TEST EVALUATION')
test_dataset = TensorDataset(testX, testY)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
probs, labels = model.predict_proba_for_auc(test_loader)

probs = np.array(probs)
labels = np.array(labels)

pred_labels = np.argmax(probs, axis=1)

# AUC
auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
auc_weighted = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')

print(f"ROC AUC (macro): {auc_macro:.4f}")
print(f"ROC AUC (weighted): {auc_weighted:.4f}")

#AUPRC
precision, recall, thresholds = precision_recall_curve(labels, probs)
auprc = average_precision_score(labels, probs)

print(f"AUPRC: {auprc:.4f}")

#F1
f1_macro = f1_score(labels, pred_labels, average='macro')
f1_weighted = f1_score(labels, pred_labels, average='weighted')

print(f"F1 Score (macro): {f1_macro:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")

num_classes = probs.shape[1]

# Binarize the true labels (one-hot)
labels_bin = label_binarize(labels, classes=np.arange(num_classes))

# AUPRC per class
auprc_per_class = []
plt.figure(figsize=(10, 6))
for c in range(num_classes):
    precision, recall, _ = precision_recall_curve(labels_bin[:, c], probs[:, c])
    ap = average_precision_score(labels_bin[:, c], probs[:, c])
    auprc_per_class.append(ap)

    plt.plot(recall, precision, label=f'Class {c} (AP = {ap:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (per class)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
