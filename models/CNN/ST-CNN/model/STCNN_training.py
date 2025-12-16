import os
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LR = 3e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/CNN/ST-CNN/best_model/stcnn_hand_posture_best.pt"

X_PART1 = "models/CNN/dataset/X_tcnn_part1.npy"
X_PART2 = "models/CNN/dataset/X_tcnn_part2.npy"
Y_FILE = "models/CNN/dataset/y_tcnn.npy"
P_FILE = "models/CNN/dataset/players.npy"

T, J, C = 120, 21, 3  # frames, joints, coords


TRAIN_PLAYERS = ["p001", "p002", "p003"]
VAL_PLAYERS   = ["p004"]
TEST_PLAYERS  = ["p005"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Residual1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return F.relu(x + out)


class SpatialTemporalNet(nn.Module):
    def __init__(self, num_classes, spatial_channels=128, temporal_channels=256):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv1d(C, spatial_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(spatial_channels, temporal_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels, 3, padding=1),
            nn.BatchNorm1d(temporal_channels),
            nn.ReLU(inplace=True),
            Residual1D(temporal_channels),
            Residual1D(temporal_channels)
        )

        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(temporal_channels, num_classes)

    def forward(self, x):
        B, T_, J_, C_ = x.shape

        x = x.reshape(B * T_, J_, C_).permute(0, 2, 1)
        x = self.spatial(x)
        x = x.view(B, T_, -1).permute(0, 2, 1)

        x = self.temporal_conv(x)
        x = self.dropout(x)

        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def accuracy_from_logits(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, total_acc, n = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits, yb) * bs
        n += bs

    return total_loss / n, total_acc / n

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy_from_logits(logits, yb) * bs
            n += bs

            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(yb.cpu().numpy())

    return (
        total_loss / n,
        total_acc / n,
        np.concatenate(preds_all),
        np.concatenate(labels_all),
    )

def main():
    
    X1 = np.load(X_PART1, mmap_mode="r")
    X2 = np.load(X_PART2, mmap_mode="r")
    X = np.concatenate([X1, X2], axis=0)

    y = np.load(Y_FILE)
    players = np.load(P_FILE)
    print("X:", X.shape, "y:", y.shape, "players:", players.shape)

    train_mask = np.isin(players, TRAIN_PLAYERS)
    val_mask   = np.isin(players, VAL_PLAYERS)
    test_mask  = np.isin(players, TEST_PLAYERS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

    train_loader = DataLoader(HandSequenceDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(HandSequenceDataset(X_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(HandSequenceDataset(X_test, y_test), BATCH_SIZE)

    model = SpatialTemporalNet(num_classes=len(np.unique(y))).to(DEVICE)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights,label_smoothing=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc, counter = 0, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:03d} | Train acc {tr_acc:.3f} | Val acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping!")
                break

    print(f"\nBest Val Acc: {best_val_acc:.3f}")

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print("\nTEST RESULTS (UNSEEN PLAYER):")
    print("Accuracy:", test_acc)
    print(classification_report(labels, preds, digits=3))
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
