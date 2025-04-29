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


param_tracker = '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/RES_HyperParamLog_Gen.csv'


# trainX1 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX1.pt')
# trainX2 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX2.pt')
# trainX3 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX3.pt')
# trainX4 = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX4.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testX.pt')

trainY1 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY0.csv')
trainY2 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY1.csv')
trainY3 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY2.csv')
trainY4 = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY3.csv')

testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testY.csv')


test = input("Test: gender/age: ")
classification = True

if test == "gender":
    trainY1 = torch.tensor(trainY1.iloc[:, 1].values)
    trainY2 = torch.tensor(trainY2.iloc[:, 1].values)
    trainY3 = torch.tensor(trainY3.iloc[:, 1].values)
    trainY4 = torch.tensor(trainY4.iloc[:, 1].values)
    testY = torch.tensor(testY.iloc[:, 1].values)
    classification = True
    criterion = nn.CrossEntropyLoss()
    param_tracker = '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/RES_HyperParamLog_Gen.csv'
elif test == "age":
    trainY1 = torch.tensor(trainY1.iloc[:, 2].values)
    trainY2 = torch.tensor(trainY2.iloc[:, 2].values)
    trainY3 = torch.tensor(trainY3.iloc[:, 2].values)
    trainY4 = torch.tensor(trainY4.iloc[:, 2].values)
    testY = torch.tensor(testY.iloc[:, 2].values)
    classification = False
    criterion = nn.MSELoss()
    param_tracker = '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/RES_HyperParamLog_Age.csv'
else:
    print("Not a valid choice")



trainY_list = [trainY1, trainY2, trainY3, trainY4]

all_trainY = torch.cat(trainY_list)

num_classes = len(torch.unique(all_trainY))

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


def performKFoldValidation(num_train_sets, trainY_list, hidden_dim1, hidden_dim2, batch_size, learning_rate, num_epochs,
                           dropout, decay):
    # Hyperparameters
    all_trainY = torch.cat(trainY_list)
    num_classes = len(torch.unique(all_trainY))

    total_acc = []
    total_time = []

    # Set one of the list sets as validation
    for i in range(num_train_sets):
        print(f'Performing kfold validation on validation set {i}')

        model = ResNetFCNN(hidden_dim1, hidden_dim2, num_classes, dropout, classification)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

        # Define Validation Set
        val_path = f'/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX{i}.pt'
        X_val_fold = torch.load(val_path)
        Y_val_fold = trainY_list[i]
        val_dataset = TensorDataset(X_val_fold, Y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Train on Remaining Train Sets
        start = time.time()

        #Loading and Training Dataset
        for j in range(num_train_sets):
            if i==j: continue
            print(f"Training on training set {j}/{num_train_sets-1}")
            train_set = torch.load(f'/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX{j}.pt')
            train_dataset = TensorDataset(train_set, trainY_list[j])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            model.train_model(train_loader, criterion, optimizer, num_epochs)

        # Validate of rermaining train list
        # FOr each fold, evaluate on validation set and send the epochs to results to list
        total_acc.append(model.evaluate_model(val_loader))
        total_time.append(time.time() - start)

    return mean(total_acc), stats.sem(total_acc), mean(total_time)


'''
    tuneModelParams
    Given trainX and trainY and a ditionary of params finds best model base on accuracy and simple model by accuracy = h1+h2

    Returns: best model {params, acc, std, time, compleixty} and simple model
'''


def tuneModelParmas(num_train_sets, trainY_list, params_dict):
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

        acc, std, time = performKFoldValidation(num_train_sets, trainY_list, h1, h2, batch_size, lr, num_epochs,
                                                               dropout, decay)

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
            'decay': decay,
            'dropout': dropout,
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

    simple = min(threshold_models, key=lambda  r: r['complexity'])

    return best_model, simple



best_model, simple_model = tuneModelParmas(4, trainY_list, params_dict)

print(f"Best Model: {best_model}")
print(f"Simple Model: {simple_model}")
