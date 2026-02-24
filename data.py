import pandas as pd
import numpy as np
from collections import defaultdict
import os

# =========================
# CONFIG
# =========================
INPUT_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\data fkp_15042025.xlsx"
OUTPUT_DIR = r"C:\Users\tavis\OneDrive\Documents\BARC\work"
TRAIN_SIZE = 10000

BINS_6D = 4   # 4 bins per dimension → 4^6 = 4096 cells
BINS_3D = 4   # fallback grid → 4^3 = 64 cells

# Position columns
POS_COLS = ["X", "Y", "Z"]

# Auto-detect rotation columns
# ROTATION_CANDIDATES = [
#     ["theta_x", "theta_y", "theta_z"],
#     ["roll", "pitch", "yaw"],
#     ["phi", "theta", "psi"],
#     ["Rx", "Ry", "Rz"],
#     ["rot_x", "rot_y", "rot_z"],
#     ["alpha", "beta", "gamma"]
# ]

ROTATION_CANDIDATES = [
    ["Thetax", "Thetay", "Thetaz"],   # ✅ your dataset
    ["theta_x", "theta_y", "theta_z"],
    ["roll", "pitch", "yaw"],
    ["phi", "theta", "psi"],
    ["Rx", "Ry", "Rz"],
    ["rot_x", "rot_y", "rot_z"],
    ["alpha", "beta", "gamma"]
]

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(INPUT_FILE)
N = len(df)

if N < TRAIN_SIZE:
    raise ValueError("Dataset smaller than 10k samples.")

# =========================
# ROTATION COLUMN DETECTION
# =========================
ROT_COLS = None
for cand in ROTATION_CANDIDATES:
    if all(c in df.columns for c in cand):
        ROT_COLS = cand
        break

if ROT_COLS is None:
    raise ValueError(f"Rotation columns not found. Available columns: {df.columns.tolist()}")

print(f"Using rotation columns: {ROT_COLS}")

ALL_OUT_COLS = POS_COLS + ROT_COLS

# =========================
# NORMALIZATION FOR BINNING
# =========================
def normalize(col):
    return (col - col.min()) / (col.max() - col.min() + 1e-12)

norm_df = df.copy()
for c in ALL_OUT_COLS:
    norm_df[c] = normalize(norm_df[c])

# =========================
# STRATIFIED GRID FUNCTION
# =========================
def stratified_sampling(norm_df, bins, dims):
    grid = defaultdict(list)

    for i, row in norm_df.iterrows():
        idx = []
        for d in dims:
            b = int(np.floor(row[d] * bins))
            if b >= bins:
                b = bins - 1
            idx.append(b)
        grid[tuple(idx)].append(i)

    return grid

# =========================
# TRY 6D GRID FIRST
# =========================
grid_6d = stratified_sampling(norm_df, BINS_6D, ALL_OUT_COLS)
occupied_cells = len(grid_6d)

print(f"6D grid occupied cells: {occupied_cells}")

USE_6D = True
if occupied_cells < 200:   # heuristic threshold
    print("6D grid too sparse → switching to 3D stratification (X,Y,Z)")
    USE_6D = False

if USE_6D:
    grid = grid_6d
    dims_used = ALL_OUT_COLS
    bins_used = BINS_6D
else:
    grid = stratified_sampling(norm_df, BINS_3D, POS_COLS)
    dims_used = POS_COLS
    bins_used = BINS_3D

# =========================
# PROPORTIONAL SAMPLING
# =========================
cell_sizes = {cell: len(idx_list) for cell, idx_list in grid.items()}
total_points = sum(cell_sizes.values())

train_indices = []

for cell, idx_list in grid.items():
    proportion = len(idx_list) / total_points
    k = int(np.round(proportion * TRAIN_SIZE))
    if k > len(idx_list):
        k = len(idx_list)

    if k > 0:
        sampled = np.random.choice(idx_list, size=k, replace=False)
        train_indices.extend(sampled.tolist())

# =========================
# FIX SIZE DRIFT
# =========================
train_indices = list(set(train_indices))

if len(train_indices) > TRAIN_SIZE:
    train_indices = np.random.choice(train_indices, TRAIN_SIZE, replace=False).tolist()
elif len(train_indices) < TRAIN_SIZE:
    remaining = list(set(range(N)) - set(train_indices))
    extra = np.random.choice(remaining, TRAIN_SIZE - len(train_indices), replace=False)
    train_indices.extend(extra.tolist())

train_indices = set(train_indices)
test_indices = set(range(N)) - train_indices

# =========================
# CREATE OUTPUT SETS
# =========================
train_df = df.loc[list(train_indices)].reset_index(drop=True)
test_df  = df.loc[list(test_indices)].reset_index(drop=True)

# =========================
# SAVE FILES
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(OUTPUT_DIR, "train_10k.xlsx")
test_path  = os.path.join(OUTPUT_DIR, "test_remaining.xlsx")

train_df.to_excel(train_path, index=False)
test_df.to_excel(test_path, index=False)

print("\nDone.")
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print(f"Saved: {train_path}")
print(f"Saved: {test_path}")
