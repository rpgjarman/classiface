import torch
import torch.nn as nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


class ResNetFCNN(nn.Module):
    def __init__(self, hidden1, hidden2, num_classes, dropout, freeze_resnet=True, classification=True):
        super(ResNetFCNN, self).__init__()
        self.classification = classification

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final classification layer of ResNet
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_resnet:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # ResNet18 outputs 512-dim feature vector
        if classification:
            self.fcnn = nn.Sequential(
                nn.Linear(512, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, num_classes)
            )
        else:
            self.fcnn = nn.Sequential(
                nn.Linear(512, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1)
            )

    def forward(self, x):
        # Input: x.shape = [batch_size, 3, 224, 224]
        features = self.feature_extractor(x)           # [batch, 512, 1, 1]
        features = features.view(features.size(0), -1) # Flatten to [batch, 512]
        return self.fcnn(features)


    def train_model(self, train_loader, criterion, optimizer, num_epochs):
        epoch_results = {}

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            correct = 0
            total_seen = 0

            print(f"\n[Epoch {epoch + 1}/{num_epochs}] Starting training...")

            for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
                optimizer.zero_grad()
                total_seen += y_batch.size(0)

                outputs = self(x_batch)

                # For Classification
                if self.classification:
                    y_batch = y_batch.long()  # for classification
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * x_batch.size(0)
                    preds = outputs.argmax(dim=1)  # pick predicted class
                    correct += (preds == y_batch).sum().item()
                # For Regression
                else:
                    outputs = outputs.squeeze()  # remove singleton dimension
                    y_batch = y_batch.float()  # ensure float for regression
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * x_batch.size(0)

                loss.backward()
                optimizer.step()

            # For Classificaiton
            if self.classification:
                avg_loss = total_loss / total_seen
                epoch_acc = correct / total_seen
                print(f"[Epoch {epoch + 1}] | Average Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.4f}")
                epoch_results[epoch + 1] = {'loss': avg_loss, 'accuracy': epoch_acc}

            # For Regression
            else:
                avg_loss = total_loss / total_seen
                print(f"[Epoch {epoch + 1}] | Average Loss (MSE): {avg_loss:.4f}")
                epoch_results[epoch + 1] = {'loss': avg_loss}

        return epoch_results

    '''
            for scores
        '''

    def predict_outputs(self, data_loader):
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                outputs = self(x_batch)

                if self.classification:
                    y_batch = y_batch.int()
                    probs = torch.softmax(outputs, dim=1)  # shape: [batch, num_classes]
                    all_preds.extend(probs[:, 1].cpu().numpy())  # class 1 probability
                    all_labels.extend(y_batch.cpu().numpy())
                else:
                    outputs = outputs.squeeze()
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

        return all_preds, all_labels

    def evaluate_model(self, test_loader):
        self.eval()
        all_probs = []
        all_labels = []
        correct = 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = self(x_batch)

                if self.classification:
                    pred = self(x_batch).argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                else:
                    outputs = outputs.squeeze()
                    all_probs.append(outputs)
                    all_labels.append(y_batch)
                    preds = torch.cat(all_probs)
                    labels = torch.cat(all_labels)

        if self.classification:
            accuracy = correct / len(test_loader.dataset)
            return accuracy
        else:
            mse = nn.functional.mse_loss(preds, labels.float())
            return mse.item()