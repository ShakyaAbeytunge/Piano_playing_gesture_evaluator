import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_FILE = "models/MLP/dataset/X_mlp.npy"
Y_FILE = "models/MLP/dataset/y_mlp.npy"
P_FILE = "models/MLP/dataset/players_mlp.npy"

TEST_PLAYERS = ["p005"] 
X = np.load(X_FILE)
y = np.load(Y_FILE)
players = np.load(P_FILE)


test_mask = np.isin(players, TEST_PLAYERS)
X_test, y_test = X[test_mask], y[test_mask]


X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)


test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)


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


INPUT_DIM = X_test.shape[1]
NUM_CLASSES = len(torch.unique(y_test))
model = MLP(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("mlp_hand_posture_best.pt", map_location=DEVICE))
model.eval()


all_preds, all_labels = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)


cm = confusion_matrix(all_labels, all_preds)
acc_per_class = cm.diagonal() / cm.sum(axis=1)

classes = ["Neutral Hands", "Wrist Flexion", "Wrist Extension", "Collapsed Knuckles", "Flat Hands"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - MLP on Test Set")

# Save as image
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

print("Per-class accuracy:", acc_per_class)
print("Confusion matrix:\n", cm)

classes = ["Good Posture", "Wrist Flexion", "Wrist Extension", "Collapsed Knuckles", "Flat Hands"]

print("Per-class accuracy and most confused class:")
for i, cls in enumerate(classes):
    row = cm[i].copy()
    row[i] = 0 
    confused_with = classes[row.argmax()] if row.sum() > 0 else "-"
    print(f"Class '{cls}' -> Accuracy: {acc_per_class[i]*100:.2f}%, Most confused with: '{confused_with}'")

overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall accuracy: {overall_acc*100:.2f}%")
print("\nConfusion matrix:\n", cm)
