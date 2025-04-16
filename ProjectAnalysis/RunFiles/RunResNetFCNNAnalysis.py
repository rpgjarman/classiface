from Models.ResNetFCNN import ResNetFCNN
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

trainX1 = torch.load(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX1.pt')
trainX2 = torch.load(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX2.pt')
trainX3 = torch.load(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX3.pt')
trainX4 = torch.load(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainX4.pt')
testX = torch.load(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testX.pt')

trainY1 = torch.tensor(pd.read_csv(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY1.csv').iloc[:,
                       1].values)
trainY2 = torch.tensor(pd.read_csv(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY2.csv').iloc[:,
                       1].values)
trainY3 = torch.tensor(pd.read_csv(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY3.csv').iloc[:,
                       1].values)
trainY4 = torch.tensor(pd.read_csv(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_trainY4.csv').iloc[:,
                       1].values)

testY = torch.tensor(pd.read_csv(
    '/Users/damienlo/Desktop/University/CS 334/Project/ModelRunFiles/ResNetFCNN/ProcessedData/RES_testY.csv').iloc[:,
                     2].values)

trainX_list = [trainX1, trainX2, trainX3, trainX4]
trainY_list = [trainY1, trainY2, trainY3, trainY4]

all_trainX = torch.cat(trainX_list)
all_trainY = torch.cat(trainY_list)

# Hyperparameters
input_dim = trainX1.shape[1]
hidden_dim1 = 256
hidden_dim2 = 128
num_classes = len(torch.unique(all_trainY))
batch_size = 64
learning_rate = 1e-3
num_epochs = 5
dropout = 0.2

model = ResNetFCNN(hidden_dim1, hidden_dim2, num_classes, dropout)

# Training
for i in range(len(trainX_list)):
    print(f'''

            Training on set {i + 1}

        ''')
    train_dataset = TensorDataset(trainX_list[i], trainY_list[i])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss Evaluator and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Train
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_seen = 0

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] Starting training...")

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            y_batch = y_batch.long()
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            # Back propergation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total_seen += y_batch.size(0)

            # Print Results every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (pred == y_batch).float().mean().item()
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Batch Accuracy: {batch_acc:.4f}")

        # Epoch Results Print
        epoch_acc = correct / total_seen
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] ✅ Average Loss: {avg_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")

    timeElapsed = time.time() - start
    print(f"Total Training Time: {timeElapsed}")

print('''

    TRAINING DONE, Begining Evaluation

    ''')

test_dataset = TensorDataset(testX, testY)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
correct = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch).argmax(dim=1)
        correct += (pred == y_batch).sum().item()

test_accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")


