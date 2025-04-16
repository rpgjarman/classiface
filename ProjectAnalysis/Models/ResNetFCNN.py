import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import torchvision.models as models

class ResNetFCNN(nn.Module):
    def __init__(self, hidden1, hidden2, num_classes, dropout, freeze_resnet=True):
        super(ResNetFCNN, self).__init__()

        # Load pretrained ResNet (you can also use resnet50, resnet101, etc.)
        resnet = models.resnet18(pretrained=True)

        # Remove the final classification layer (fc)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # all layers except the final FC

        # Optionally freeze resnet weights
        if freeze_resnet:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # ResNet18 outputs 512-dim feature vector
        self.fcnn = nn.Sequential(
            nn.Linear(512, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        # Input: x.shape = [batch_size, 3, 224, 224]
        features = self.feature_extractor(x)           # [batch, 512, 1, 1]
        features = features.view(features.size(0), -1) # Flatten to [batch, 512]
        return self.fcnn(features)


    def train_model(self, train_loader_list, criterion, optimizer, num_epochs):
        for i in range(len(train_loader_list)):
            train_loader = train_loader_list[i]
            print(f'Training on dataset {i}')
            for epoch in range(num_epochs):
                self.train()
                total_loss = 0
                correct = 0
                total_seen = 0

                print(f"\n[Epoch {epoch + 1}/{num_epochs}] Starting training...")

                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    y_batch = y_batch.long()
                    optimizer.zero_grad()
                    outputs = self(x_batch)
                    loss = criterion(outputs, y_batch)
                    # Back propergation
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total_seen += y_batch.size(0)

                    # Print Results every 10 batches
                    # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    #     batch_acc = (pred == y_batch).float().mean().item()
                    #     print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                    #           f"Loss: {loss.item():.4f} | Batch Accuracy: {batch_acc:.4f}")

                # Epoch Results Print
                epoch_acc = correct / total_seen
                avg_loss = total_loss / len(train_loader)
                print(f"[Epoch {epoch + 1}] | Average Loss: {avg_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")

    def predict_proba_for_auc(self, test_loader):
        """
        Returns predicted probabilities for class 1 and true labels — for AUC computation.
        """
        self.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = self(x_batch)
                probs = torch.softmax(outputs, dim=1)  # shape: [batch, num_classes]
                all_probs.extend(probs[:, 1].cpu().numpy())  # class 1 probability
                all_labels.extend(y_batch.cpu().numpy())

        return all_probs, all_labels

    def evaluate_model(self, test_loader):
        self.eval()
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                pred = self(x_batch).argmax(dim=1)
                correct += (pred == y_batch).sum().item()

        return correct / len(test_loader.dataset)