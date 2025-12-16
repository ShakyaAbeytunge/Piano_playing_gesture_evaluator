from torch.utils.data import DataLoader
from models.STGCN.STGCN_model import STGCN, GraphDataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 1)
    return A

# ---------------- LOAD DATASET ----------------
DATA_ROOT = "dataset"
PLAYER_FOLDER = "p005"
val_folder = os.path.join(DATA_ROOT, PLAYER_FOLDER)

test_files = sorted([os.path.join(val_folder, f) for f in os.listdir(val_folder) if f.endswith(".npz")])

test_loader = DataLoader(GraphDataset(test_files), batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = build_adjacency(num_nodes=21, edges=HAND_EDGES)

model_path = "models/STGCN/best_models/best_model_0.7246_4sec.pth"
model = STGCN(num_classes=5, A=A).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.item())
        all_labels.append(y.item())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
acc_per_class = cm.diagonal() / cm.sum(axis=1)

classes = ["Neutral Hands", "Wrist Flexion", "Wrist Extension", "Collapsed Knuckles", "Flat Hands"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - ST-GCN on Test Set")

# Save as image
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

print("Per-class accuracy:", acc_per_class)
print("Confusion matrix:\n", cm)

classes = ["Good Posture", "Wrist Flexion", "Wrist Extension", "Collapsed Knuckles", "Flat Hands"]

# print("\\begin{tabular}{lcc}")
# print("\\hline")
# print("Class & Accuracy & Most confused with \\\\")
# print("\\hline")
# for i, cls in enumerate(classes):
#     # Find the class it's most confused with
#     row = cm[i]
#     row[i] = 0  # ignore correct predictions
#     if row.sum() == 0:
#         confused_with = "-"
#     else:
#         confused_with = classes[row.argmax()]
#     print(f"{cls} & {acc_per_class[i]*100:.1f}\\% & {confused_with} \\\\")
# print("\\hline")
# print("\\end{tabular}")

for i, cls in enumerate(classes):
    # Find the class it's most confused with
    row = cm[i]
    row[i] = 0  # ignore correct predictions
    if row.sum() == 0:
        confused_with = "-"
    else:
        confused_with = classes[row.argmax()]
    print(f"Class '{cls}' is most confused with: '{confused_with}'. Accuracy: {acc_per_class[i]*100:.2f}%")
print(f"Overall accuracy: {accuracy_score(all_labels, all_preds)*100:.2f}%")

