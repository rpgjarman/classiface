import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
from sklearn.model_selection import KFold
from Model import FCNN
import time
from statistics import mean
from scipy import stats
import itertools
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm




trainX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainX.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testX.pt')

trainY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainY.csv')
testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testY.csv')

param_tracker = '/Users/damienlo/Desktop/University/CS 334/Project/ProjectAnalysis/HyperParamTuneTracker.csv'

trainY = torch.tensor(trainY.iloc[:, 0].values)
testY = torch.tensor(testY.iloc[:, 0].values)

folds = 5

params_dict = {
    'lr': [1e-2, 1e-3, 1e-4],
    'hidden_dim1': [512, 256, 128],
    'hidden_dim2': [128, 64, 32],
    'batch_size': [32, 64, 128],
    'num_epochs': [10, 20, 50, 100]
}

print(params_dict)


print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

'''
    performKFoldValidation
    Given a model and parameters and number of folds, performs, kfold validaiton on train set
    
    Returns: Avereage accuarcy and time for training+validate across all folds
'''
def performKFoldValidation(trainX, trainY, k_fold, hidden_dim1, hidden_dim2, batch_size, learning_rate, num_epochs):

    # Hyperparameters
    input_dim = trainX.shape[1]
    num_classes = len(torch.unique(trainY))



    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    total_acc = []
    total_time = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainX)):
        print(f"Validating fold: {fold}")
        X_train_fold = trainX[train_idx]
        X_val_fold = trainX[val_idx]
        y_train_fold = trainY[train_idx]
        y_val_fold = trainY[val_idx]

        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)

        # Setting Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Defining Model
        model = FCNN(input_dim, hidden_dim1, hidden_dim2, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()


        # Training Model And Testing On Fold
        start = time.time()
        best_epoch, best_accuracy = model.train_model(train_loader,criterion,optimizer,num_epochs)

        # Logging Average Metrics
        total_acc.append(model.evaluate_model(val_loader))
        total_time.append(time.time() - start)

    return mean(total_acc),stats.sem(total_acc), mean(total_time)

'''
    tuneModelParams
    Given trainX and trainY and a ditionary of params finds best model base on accuracy and simple model by accuracy = h1+h2
    
    Returns: best model {params, acc, std, time, compleixty} and simple model
'''
def tuneModelParmas(trainX,trainY,params_dict):
    print("BEGINING PARAMETER TUNING")
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    results = []

    # All combinations
    all_combinations = list(itertools.product(*values))
    print(f"Total Number of Parameter combinations is: {all_combinations}")

    for combo in all_combinations:
        param_combo = dict(zip(keys, combo))


        print("=============================")
        print("TESTING PARAMETERS:")
        print(f"{param_combo}")

        lr = param_combo['lr']
        h1 = param_combo['hidden_dim1']
        h2 = param_combo['hidden_dim2']
        batch_size = param_combo['batch_size']
        num_epochs = param_combo['num_epochs']

        acc, std, time = performKFoldValidation(trainX, trainY, 5, h1, h2, batch_size, lr, num_epochs)

        combo_result = {
            'params': param_combo,
            'acc': acc,
            'std': std,
            'time': time,
            'complexity': h1 + h2,  # You can change this metric if you like
        }

        results.append(combo_result)
        pd.DataFrame([{
            "lr" : lr,
            "h1" : h1,
            "h2" : h2,
            "batch_size" : batch_size,
            "num_epochs" : num_epochs,
            "acc" : acc,
            "std" : std,
            "time" : time,
            "complexity" : h1 + h2,
        }]).to_csv(param_tracker, index=False, header=False)



        print("PARAMETER COMBINATION TEST COMPLETED")
        print("=============================")

    print("TUNING COMPLETED")
    best_model = max(results, key=lambda r: r['acc'])
    threshold = best_model['acc'] - best_model['std']  # within 1 SD BELOW the best

    # Filter all models within 1 SD of the best
    threshold_models = [r for r in results if r['acc'] >= threshold]

    # Among those, pick the one with the lowest complexity
    simple = min(threshold_models, key=lambda r: r['complexity'])

    return best_model, simple


print("TEST")
best_model, simple_model = tuneModelParmas(trainX,trainY,params_dict)

print(f"Best Model: {best_model}")
print(f"Simple Model: {simple_model}")






