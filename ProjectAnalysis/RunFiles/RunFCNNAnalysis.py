import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

trainX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainX.pt')
testX = torch.load('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testX.pt')

trainY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/trainY.csv')
testY = pd.read_csv('/Users/damienlo/Desktop/University/CS 334/Project/Datasets/ProcessedData/testY.csv')

trainY = torch.tensor(trainY.iloc[:, 2].values)
testY = torch.tensor(testY.iloc[:, 2].values)


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

# Define Model
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes, p=dropout):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = FCNN(input_dim, hidden_dim1, hidden_dim2, num_classes)




# Data Loaders
train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)
# Split into Batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Loss Evaluator and Optimiser
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train
start = time.time()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total_seen = 0

    print(f"\n[Epoch {epoch+1}/{num_epochs}] Starting training...")

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
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Batch Accuracy: {batch_acc:.4f}")

    # Epoch Results Print
    epoch_acc = correct / total_seen
    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] ✅ Average Loss: {avg_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")

timeElapsed = time.time() - start
print(f"Total Training Time: {timeElapsed}")

# Test Accuracy Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch).argmax(dim=1)
        correct += (pred == y_batch).sum().item()

test_accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

