"""
improved_tcnn_clean.py

Spatial-Temporal CNN for hand posture classification from keypoint sequences.

Files required:
    - X_tcnn.npy   (N, T=30, J=21, C=3)
    - y_tcnn.npy   (N,)

Usage:
    python improved_tcnn_clean.py
"""

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config
# ---------------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "tcnn_hand_posture_best.pt"

X_FILE = "X_tcnn.npy"
Y_FILE = "y_tcnn.npy"

T, J, C = 30, 21, 3  # frames, joints, coords

# ---------------------------
# Helpers: reproducibility
# ---------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ---------------------------
# Dataset
# ---------------------------
class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Model
# ---------------------------
class Residual1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)

class SpatialTemporalNet(nn.Module):
    def __init__(self, num_classes, spatial_channels=128, temporal_channels=256):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(C, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, spatial_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(spatial_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.temporal_conv1 = nn.Conv1d(spatial_channels, temporal_channels, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(temporal_channels)
        self.resblock1 = Residual1D(temporal_channels)
        self.resblock2 = Residual1D(temporal_channels)
        self.temporal_conv2 = nn.Conv1d(temporal_channels, temporal_channels, kernel_size=3, padding=1)
        self.temporal_bn2 = nn.BatchNorm1d(temporal_channels)
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(temporal_channels, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T_, J_, C_ = x.shape
        x_sp = x.reshape(B * T_, J_, C_).permute(0, 2, 1)  # (B*T, C, J)
        x_sp = self.spatial(x_sp)                           # (B*T, spatial_channels, 1)
        x_sp = x_sp.view(B, T_, -1)                         # (B, T, spatial_channels)
        x_temp = x_sp.permute(0, 2, 1)                      # (B, spatial_channels, T)
        x_temp = F.relu(self.temporal_bn1(self.temporal_conv1(x_temp)))
        x_temp = self.resblock1(x_temp)
        x_temp = self.resblock2(x_temp)
        x_temp = F.relu(self.temporal_bn2(self.temporal_conv2(x_temp)))
        x_temp = self.dropout(x_temp)
        x_out = self.pool(x_temp).squeeze(-1)               # (B, temporal_channels)
        logits = self.fc(x_out)
        return logits

# ---------------------------
# Training / Evaluation utils
# ---------------------------
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, yb) * batch_size
        n += batch_size
    return running_loss / n, running_acc / n

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(logits, yb) * batch_size
            n += batch_size
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    return running_loss/n, running_acc/n, np.concatenate(all_preds), np.concatenate(all_labels)

# ---------------------------
# Main
# ---------------------------
def main():
    print("Device:", DEVICE)
    assert os.path.exists(X_FILE) and os.path.exists(Y_FILE), "Missing X_tcnn.npy or y_tcnn.npy"

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    print("X shape:", X.shape, "y shape:", y.shape)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    train_loader = DataLoader(HandSequenceDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HandSequenceDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(np.unique(y))
    model = SpatialTemporalNet(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    patience, counter = 10, 0

    print("Start training...")
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch:03d} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved (val_acc={val_acc:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"No improvement for {patience} epochs. Early stopping!")
                break

    print(f"Training finished. Best val acc: {best_val_acc:.4f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    _, test_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
    print("\nValidation final report:")
    print(classification_report(labels, preds, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
