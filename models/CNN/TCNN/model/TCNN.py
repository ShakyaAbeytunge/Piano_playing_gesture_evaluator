import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TemporalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=21*3, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B = x.size(0)
        T, J, C = x.size(1), x.size(2), x.size(3) 
        x = x.reshape(B, T, J*C)                    
        x = x.transpose(1, 2)       
        x = self.temporal_conv(x)
        x = self.dropout(x)
        x = x.squeeze(-1)
        return self.fc(x)


X = np.load("models/CNN/dataset/X_tcnn.npy")
y = np.load("models/CNN/dataset/y_tcnn.npy")
players = np.load("models/CNN/dataset/players.npy")  

TRAIN_PLAYERS = ["p001", "p002", "p003"]
VAL_PLAYERS   = ["p004"]
TEST_PLAYERS  = ["p005"]

train_mask = np.isin(players, TRAIN_PLAYERS)
val_mask   = np.isin(players, VAL_PLAYERS)
test_mask  = np.isin(players, TEST_PLAYERS)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val     = X[val_mask], y[val_mask]
X_test, y_test   = X[test_mask], y[test_mask]

train_loader = DataLoader(HandSequenceDataset(X_train, y_train), batch_size=8, shuffle=True)
val_loader   = DataLoader(HandSequenceDataset(X_val, y_val), batch_size=8)
test_loader  = DataLoader(HandSequenceDataset(X_test, y_test), batch_size=8)

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

num_classes = len(np.unique(y))
model = TemporalCNN(num_classes=num_classes).to(DEVICE)


criterion = nn.CrossEntropyLoss( label_smoothing=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

EPOCHS = 1000
PATIENCE = 20
best_val_loss = float('inf')
counter = 0


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            val_loss += criterion(logits, yb).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:03d} | Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "models/CNN/TCNN/best_model/tcnn_hand_posture_best.pt")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping ")
            break

model.load_state_dict(torch.load("models/CNN/TCNN/best_model/tcnn_hand_posture_best.pt"))
model.eval()

def evaluate(loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            predicted = logits.argmax(dim=1)
            correct += (predicted == yb).sum().item()
            total += len(yb)
    return correct / total

val_acc = evaluate(val_loader)
test_acc = evaluate(test_loader)

print(f"\nValidation Accuracy: {val_acc:.3f}")
print(f"Test Accuracy : {test_acc:.3f}")
