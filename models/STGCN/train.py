import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.STGCN_model import STGCN, GraphDataset
import argparse


DATA_ROOT = "dataset"

all_players = ["p001","p002", "p003", "p004","p005", "p006"]
player_files = {p: sorted(glob.glob(os.path.join(DATA_ROOT, p, "*.npz"))) for p in all_players}

# ---------------- Graph ----------------
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

# ---------------- Training ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = build_adjacency(21, HAND_EDGES)

num_epochs = 80
batch_size = 8
num_classes = 5

cv_val_accs = []

for val_player in all_players:
    print(f"=== CV Fold: val_player={val_player} ===")
    
    # Split files
    val_files = player_files[val_player]
    train_files = [f for p in all_players if p != val_player for f in player_files[p]]

    train_loader = DataLoader(GraphDataset(train_files), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GraphDataset(val_files), batch_size=batch_size, shuffle=False)

    # Initialize model for each fold
    model = STGCN(num_classes=num_classes, A=A).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Early stopping parameters
    patience = 20         # stop if val_acc doesn't improve for 10 epochs
    no_improve_count = 0   # counter
    best_val_acc = 0       # best validation accuracy so far

    for epoch in range(num_epochs):
        # ---------------- Train ----------------
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        # ---------------- Validation ----------------
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                pred = out.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        # ---------------- Early Stopping ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), f"best_model_{val_player}.pth")
        else:
            no_improve_count += 1

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}, best val acc={best_val_acc:.4f}")
            break


    cv_val_accs.append(best_val_acc)
    print(f"Best Val Acc for {val_player}: {best_val_acc:.4f}")

print("=== Cross-Validation Summary ===")
for p, acc in zip(all_players, cv_val_accs):
    print(f"{p}: {acc:.4f}")
print(f"Mean CV Val Acc: {np.mean(cv_val_accs):.4f}")