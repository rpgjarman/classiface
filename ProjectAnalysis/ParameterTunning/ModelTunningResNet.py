from Models.ResNetFCNN import ResNetFCNN
import torch.nn as nn
import torch
import pandas as pd
import time
from statistics import mean
from scipy import stats
import itertools
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import random


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

folds = 5

params_dict = {
    'lr': [1e-2, 1e-3, 1e-4],
    'hidden_dim1': [512, 256, 128],
    'hidden_dim2': [128, 64, 32],
    'batch_size': [32, 64, 128],
    'dropout': [0.1, 0.2, 0.3],
    'decay': [1e-4, 1e-5, 1e-6],
    'num_epochs': [10,20]
}

'''
    performKFoldValidation
    Given a model and parameters and number of folds, performs, kfold validaiton on train set with 5 folds

    Returns: Avereage accuarcy and time for training+validate across all folds
'''


def performKFoldValidation(trainX_list, trainY_list, hidden_dim1, hidden_dim2, batch_size, learning_rate, num_epochs,
                           dropout, decay):
    # Hyperparameters
    all_trainY = torch.cat(trainY_list)
    num_classes = len(torch.unique(all_trainY))

    total_acc = []
    total_time = []

    for i in range(len(trainX_list)):
        train_loader_list = []

        print(f'Performing kfold validation on validation set {i}')

        for j in range(len(trainX_list)):
            if i==j: break
            train_dataset = TensorDataset(trainX_list[j], trainY_list[j])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_loader_list.append(train_loader)

        X_val_fold = trainX_list[i]
        Y_val_fold = trainY_list[i]
        val_dataset = TensorDataset(X_val_fold, Y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)


        model = ResNetFCNN(hidden_dim1, hidden_dim2, num_classes, dropout)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        model.train_model(train_loader_list, criterion, optimizer, num_epochs)

        total_acc.append(model.evaluate_model(val_loader))
        total_time.append(time.time() - start)

        total_acc.append(model.evaluate_model(val_loader))
        total_time.append(time.time() - start)

    return mean(total_acc), stats.sem(total_acc), mean(total_time)


'''
    tuneModelParams
    Given trainX and trainY and a ditionary of params finds best model base on accuracy and simple model by accuracy = h1+h2

    Returns: best model {params, acc, std, time, compleixty} and simple model
'''


def tuneModelParmas(trainX_list, trainY_list, params_dict):
    print("BEGINING PARAMETER TUNING")
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    results = []

    # All combinations
    all_combinations = list(itertools.product(*values))
    random.shuffle(all_combinations)

    print(f"Total Number of Parameter combinations is: {len(all_combinations)}")

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
        dropout = param_combo['dropout']
        decay = param_combo['decay']

        acc, std, time = performKFoldValidation(trainX_list, trainY_list, h1, h2, batch_size, lr, num_epochs,
                                                               dropout, decay)

        epoch_results = str(epoch_results)

        combo_result = {
            'params': param_combo,
            'acc': acc,
            'std': std,
            'time': time,
            'complexity': h1 + h2,
        }

        results.append(combo_result)
        pd.DataFrame([{
            "lr": lr,
            "h1": h1,
            "h2": h2,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            'epoch_acc': epoch_results,
            'dropout': dropout,
            'decay': decay,
            "acc": acc,
            "std": std,
            "time": time,
            "complexity": h1 + h2,
        }]).to_csv(param_tracker, mode='a', index=False, header=False)

        print("PARAMETER COMBINATION TEST COMPLETED")
        print(f'Results: {combo_result}')

        print("=============================")

    print("TUNING COMPLETED")
    best_model = max(results, key=lambda r: r['acc'])
    threshold = best_model['acc'] - best_model['std']

    # Filter all models within 1 SD of the best
    threshold_models = [r for r in results if r['acc'] >= threshold]

    simple = min(threshold_models, key=lambda r: r['complexity'])

    return best_model, simple


print("TEST")
best_model, simple_model = tuneModelParmas(trainX_list, trainY_list, params_dict)

print(f"Best Model: {best_model}")
print(f"Simple Model: {simple_model}")
