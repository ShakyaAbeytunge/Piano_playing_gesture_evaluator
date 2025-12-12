import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
X = np.load("X_mlp.npy")
y = np.load("y_mlp.npy")

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)


    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=63, num_classes=len(torch.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(3000):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss {total_loss:.3f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == yb).sum().item()
        total += len(yb)

print(f"Test accuracy: {correct/total:.2f}")

torch.save(model.state_dict(), "mlp_hand_posture.pt")
