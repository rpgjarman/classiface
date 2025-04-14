import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

class FCNN(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, num_classes):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)



    def train_model(self, train_loader, criterion, optimizer, num_epochs):
        best_epoch = 0
        best_accuracy = 0

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            correct = 0
            total_seen = 0
            print(f"\n[Epoch {epoch+1}/{num_epochs}] Starting training...")

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total_seen += y_batch.size(0)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    batch_acc = (pred == y_batch).float().mean().item()
                    # print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                    #       f"Loss: {loss.item():.4f} | Batch Accuracy: {batch_acc:.4f}")

            epoch_acc = correct / total_seen
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] ✅ Avg Loss: {avg_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_epoch = epoch

        return best_epoch, best_accuracy

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                pred = self(x_batch).argmax(dim=1)
                correct += (pred == y_batch).sum().item()

        return correct / len(test_loader.dataset)