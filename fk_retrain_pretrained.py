import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib
import os

# =========================
# CONFIG
# =========================
# TRAIN_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\train_10k.xlsx"
# TEST_FILE  = r"C:\Users\tavis\OneDrive\Documents\BARC\work\test_remaining.xlsx"
TRAIN_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\train_10k.xlsx"
TEST_FILE  = r"C:\Users\tavis\OneDrive\Documents\BARC\work\test_remaining.xlsx"
OLD_MODEL_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\fk_model_best.pth"

OUTPUT_MODEL_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\fk_model_full_best.pth"
OUTPUT_XSCALER = r"C:\Users\tavis\OneDrive\Documents\BARC\work\x_scaler_full.pkl"
OUTPUT_YSCALER = r"C:\Users\tavis\OneDrive\Documents\BARC\work\y_scaler_full.pkl"

INPUT_COLS  = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]
OUTPUT_COLS = ["X","Y","Z","Thetax","Thetay","Thetaz"]

BATCH_SIZE = 256
EPOCHS = 800
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# LOAD + MERGE DATASETS
# =========================
print("Loading datasets...")
train_df = pd.read_excel(TRAIN_FILE)
test_df  = pd.read_excel(TEST_FILE)

full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

print("Total samples:", len(full_df))

X = full_df[INPUT_COLS].values.astype(np.float32)
Y = full_df[OUTPUT_COLS].values.astype(np.float32)

# =========================
# NORMALIZATION (FULL DATA)
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

full_ds = FKDataset(Xn, Yn)
full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)

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
# LOAD OLD MODEL WEIGHTS
# =========================
if os.path.exists(OLD_MODEL_PATH):
    print("Loading pretrained model weights...")
    model.load_state_dict(torch.load(OLD_MODEL_PATH, map_location=DEVICE))
else:
    print("Pretrained model not found. Training from scratch.")

# =========================
# LOSS
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
# TRAINING LOOP (FULL DATA FINETUNING)
# =========================
best_loss = 1e12

for epoch in range(EPOCHS):
    model.train()
    tl = 0

    for xb, yb in full_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        pred = model(xb)
        loss = fk_loss(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tl += loss.item()

    tl /= len(full_loader)

    if tl < best_loss:
        best_loss = tl
        torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Train Loss {tl:.8f}")

print("\nFull-dataset training complete.")
print("Best model saved as:", OUTPUT_MODEL_PATH)

# =========================
# SAVE NEW SCALERS
# =========================
joblib.dump(x_scaler, OUTPUT_XSCALER)
joblib.dump(y_scaler, OUTPUT_YSCALER)

print("Scalers saved:")
print(OUTPUT_XSCALER)
print(OUTPUT_YSCALER)
