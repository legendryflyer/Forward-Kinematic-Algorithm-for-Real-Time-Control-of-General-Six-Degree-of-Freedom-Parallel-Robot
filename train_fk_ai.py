import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
TRAIN_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\train_10k.xlsx"
TEST_FILE  = r"C:\Users\tavis\OneDrive\Documents\BARC\work\test_remaining.xlsx"

INPUT_COLS  = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]
OUTPUT_COLS = ["X","Y","Z","Thetax","Thetay","Thetaz"]

BATCH_SIZE = 128
EPOCHS = 500
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
train_df = pd.read_excel(TRAIN_FILE)

X = train_df[INPUT_COLS].values.astype(np.float32)
Y = train_df[OUTPUT_COLS].values.astype(np.float32)

# =========================
# NORMALIZATION
# =========================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

Xn = x_scaler.fit_transform(X)
Yn = y_scaler.fit_transform(Y)

# =========================
# DATASET
# =========================
class FKDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

Xtr, Xval, Ytr, Yval = train_test_split(Xn, Yn, test_size=0.15, random_state=42)

train_ds = FKDataset(Xtr, Ytr)
val_ds   = FKDataset(Xval, Yval)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
class FKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6,128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128,256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256,256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128,64),
            nn.GELU(),

            nn.Linear(64,6)
        )

    def forward(self, x):
        return self.net(x)

model = FKNet().to(DEVICE)

# =========================
# LOSS (weighted SE(3))
# =========================
trans_w = 1.0
rot_w   = 1.0

huber = nn.SmoothL1Loss()

def fk_loss(pred, target):
    t_loss = huber(pred[:,0:3], target[:,0:3])
    r_loss = huber(pred[:,3:6], target[:,3:6])
    return trans_w*t_loss + rot_w*r_loss

# =========================
# OPTIMIZER
# =========================
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

# =========================
# TRAINING LOOP
# =========================
best_val = 1e9

for epoch in range(EPOCHS):
    model.train()
    tl = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        pred = model(xb)
        loss = fk_loss(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tl += loss.item()

    model.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = fk_loss(pred, yb)
            vl += loss.item()

    tl /= len(train_loader)
    vl /= len(val_loader)

    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), "fk_model_best.pth")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Train {tl:.6f} | Val {vl:.6f}")

print("Training complete. Best model saved as fk_model_best.pth")

# =========================
# SAVE SCALERS
# =========================
import joblib
joblib.dump(x_scaler, "x_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")