import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

class GraphDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in '{folder}'. Verify the path and data split."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"]).long()
        return X, y

DATA_ROOT = "dataset_graphs"
batch_size = 8

train_ds = GraphDataset(os.path.join(DATA_ROOT, "train"))
# val_ds   = GraphDataset(os.path.join(DATA_ROOT, "val"))
test_ds  = GraphDataset(os.path.join(DATA_ROOT, "test"))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(val_ds, batch_size=batch_size)
test_loader  = DataLoader(test_ds, batch_size=batch_size)

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def build_adjacency(num_nodes, edges):
    A = np.zeros((num_nodes, num_nodes))
    for i,j in edges:
        A[i,j] = 1
        A[j,i] = 1
    np.fill_diagonal(A, 1)
    return A

# ---------------- ST-GCN Block ----------------
class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, dropout=0.3):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        # Graph convolution (learnable)
        self.gcn = nn.Conv2d(in_c, out_c, kernel_size=1)

        # Temporal convolution
        self.tcn = nn.Conv2d(
            out_c, out_c,
            kernel_size=(3, 1),
            padding=(1, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual
        if in_c != out_c or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.res = nn.Identity()

    def forward(self, x):
        # x: (N, C, T, V)

        res = self.res(x)

        # Graph aggregation
        x = torch.einsum("nctv,vw->nctw", x, self.A)

        # Learnable spatial mixing
        x = self.gcn(x)

        # Temporal modeling
        x = self.tcn(x)
        x = self.bn(x)
        x = self.drop(x)

        return self.relu(x + res)

# ---------------- ST-GCN ----------------
class STGCN(nn.Module):
    def __init__(self, num_classes, A):
        super().__init__()
        self.layer1 = STGCNBlock(3, 8, A)
        self.layer2 = STGCNBlock(8, 16, A)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

A = build_adjacency(21, HAND_EDGES)

model = STGCN(num_classes=5, A=A).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 100

best_val_loss = float('inf')

for epoch in range(num_epochs):
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}", end=' ')
    val_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(test_loader)}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → Saved best model at epoch {epoch+1}")
    
model.load_state_dict(torch.load("best_model.pth"))
correct = 0 
test_loss = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
print(f"Test Loss: {test_loss/len(test_loader)}, Test Accuracy: {correct/len(test_ds)}")