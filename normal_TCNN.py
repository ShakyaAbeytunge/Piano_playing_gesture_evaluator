import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# DATASET CLASS
# ------------------------------
class HandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 30, 21, 3)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------
# TEMPORAL CNN MODEL
# ------------------------------
class TemporalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(63, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, 30, -1)        # (B, 30, 63)
        x = x.transpose(1, 2)           # (B, 63, 30)
        x = self.temporal_conv(x)       # (B, 256, 1)
        x = x.squeeze(-1)               # (B, 256)
        return self.fc(x)


# ------------------------------
# LOAD DATA
# ------------------------------
print("Loading dataset...")
X = np.load("X_tcnn.npy")
y = np.load("y_tcnn.npy")

dataset = HandSequenceDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

print(f"Train samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")

# ------------------------------
# TRAINING SETUP
# ------------------------------
num_classes = len(np.unique(y))
model = TemporalCNN(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-3)

EPOCHS = 1000
PATIENCE = 20  # Early stopping patience
best_val_loss = float('inf')
counter = 0

# ------------------------------
# TRAINING LOOP WITH EARLY STOPPING
# ------------------------------
print("\nTraining TCNN with Early Stopping...")

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Validate ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb_val, yb_val in val_loader:
            xb_val, yb_val = xb_val.to(DEVICE), yb_val.to(DEVICE)
            pred_val = model(xb_val)
            loss_val = criterion(pred_val, yb_val)
            val_loss += loss_val.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss={total_loss:.3f}, Val Loss={val_loss:.3f}")

    # --- Early Stopping Check ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "tcnn_hand_posture_best.pt")
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Early stopping!")
            break

# ------------------------------
# EVALUATION
# ------------------------------
model.load_state_dict(torch.load("tcnn_hand_posture_best.pt"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        predicted = pred.argmax(dim=1)
        correct += (predicted == yb).sum().item()
        total += len(yb)

accuracy = correct / total
print("\nValidation Accuracy:", accuracy)
