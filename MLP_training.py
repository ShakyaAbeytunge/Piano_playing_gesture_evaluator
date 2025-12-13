import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ------------------------------
# Load dataset
# ------------------------------
X = np.load("X_mlp.npy")
y = np.load("y_mlp.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ------------------------------
# MLP model
# ------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)

model = MLP(input_dim=63, num_classes=len(torch.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-5)

# ------------------------------
# Training with early stopping
# ------------------------------
EPOCHS = 3000
patience = 40  # stop if no improvement for 30 epochs
best_val_loss = float('inf')
counter = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.3f}, Val Loss: {val_loss:.3f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "mlp_hand_posture_best.pt")  # save best model
    else:
        counter += 1
        if counter >= patience:
            print(f"No improvement for {patience} epochs. Early stopping!")
            break

# ------------------------------
# Evaluation
# ------------------------------
model.load_state_dict(torch.load("mlp_hand_posture_best.pt"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in val_loader:
        pred = model(xb)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == yb).sum().item()
        total += len(yb)

print(f"Validation accuracy: {correct/total:.2f}")
