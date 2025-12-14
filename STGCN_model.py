import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------- Dataset ----------------
class GraphDataset(Dataset):
    def __init__(self, files):
        # files: list of .npz paths
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError("No .npz files provided to GraphDataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        X = torch.tensor(data["X"], dtype=torch.float32)  # (C,T,V)
        y = torch.tensor(data["y"]).long()
        return X, y

# ---------------- ST-GCN Block ----------------
class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=2, dropout=0):
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