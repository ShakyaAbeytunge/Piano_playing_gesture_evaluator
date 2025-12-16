import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import random


SEED = 42
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_FILE = "models/MLP/dataset/X_mlp.npy"
Y_FILE = "models/MLP/dataset/y_mlp.npy"
P_FILE = "models/MLP/dataset/players_mlp.npy"

INPUT_DIM = 252 # 21 joints × 3 coords × 4 stats
NUM_CLASSES = 5

TRAIN_PLAYERS = ["p001", "p002", "p003", "p004", "p006"]
VAL_PLAYERS   = ["p002"] 
TEST_PLAYERS  = ["p005"]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

X = np.load(X_FILE)
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

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    ),
    batch_size=BATCH_SIZE
)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(INPUT_DIM, NUM_CLASSES).to(DEVICE)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights ,label_smoothing=0.1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

best_val_loss = float("inf")
counter = 0

for epoch in range(1, EPOCHS + 1):
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
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

    val_loss /= len(val_loader)
    val_acc = correct / total if total > 0 else 0

    print(f"Epoch {epoch:03d} | Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f} | Val Acc {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "models/MLP/best_model/mlp_hand_posture_best.pt")
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early stopping")
            break
        
model.load_state_dict(torch.load("models/MLP/best_model/mlp_hand_posture_best.pt.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\nTEST RESULTS (UNSEEN PLAYER):")
print(classification_report(all_labels, all_preds, digits=3))
print(confusion_matrix(all_labels, all_preds))
