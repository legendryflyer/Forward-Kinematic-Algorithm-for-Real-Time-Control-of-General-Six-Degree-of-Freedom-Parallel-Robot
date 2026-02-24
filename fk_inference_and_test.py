import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
# MODEL_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\fk_model_best.pth"
# XSCALER_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\x_scaler.pkl"
# YSCALER_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\y_scaler.pkl"

MODEL_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\fk_model_full_best.pth"
XSCALER_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\x_scaler_full.pkl"
YSCALER_PATH = r"C:\Users\tavis\OneDrive\Documents\BARC\work\y_scaler_full.pkl"

# TEST_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\test_remaining.xlsx"

INPUT_COLS  = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]
OUTPUT_COLS = ["X","Y","Z","Thetax","Thetay","Thetaz"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

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

# =========================
# LOAD MODEL + SCALERS
# =========================
model = FKNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

x_scaler = joblib.load(XSCALER_PATH)
y_scaler = joblib.load(YSCALER_PATH)

# print("Model and scalers loaded.")

# =========================
# TEST ON REMAINING DATASET
# =========================
# print("\n--- Evaluating on test_remaining.xlsx ---")

# test_df = pd.read_excel(TEST_FILE)

# Xt = test_df[INPUT_COLS].values.astype(np.float32)
# Yt = test_df[OUTPUT_COLS].values.astype(np.float32)

# Xt_n = x_scaler.transform(Xt)
# Xt_n = torch.tensor(Xt_n, dtype=torch.float32).to(DEVICE)

# with torch.no_grad():
#     pred_n = model(Xt_n).cpu().numpy()

# pred = y_scaler.inverse_transform(pred_n)

# # Errors
# trans_err = np.linalg.norm(pred[:,0:3] - Yt[:,0:3], axis=1)
# rot_err   = np.linalg.norm(pred[:,3:6] - Yt[:,3:6], axis=1)

# print("\n--- Accuracy Report ---")
# print("Mean translation error (mm):", np.mean(trans_err))
# print("Max translation error (mm):", np.max(trans_err))
# print("Median translation error (mm):", np.median(trans_err))

# print("Mean rotation error:", np.mean(rot_err))
# print("Max rotation error:", np.max(rot_err))
# print("Median rotation error:", np.median(rot_err))

# print("\nBelow thresholds:")
# print(" < 0.1 mm translation:", np.mean(trans_err < 0.1)*100, "%")
# print(" < 0.1 rotation:", np.mean(rot_err < 0.1)*100, "%")

# =========================
# MANUAL INPUT MODE
# =========================
def manual_fk_predict():
    print("\n--- Manual FK Prediction Mode ---")
    print("Enter 6 leg lengths (space separated):")
    print("Format: L1 L2 L3 L4 L5 L6")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Leg lengths > ")

        if user_input.lower().strip() == "exit":
            break

        try:
            legs = list(map(float, user_input.strip().split()))
            if len(legs) != 6:
                print("Please enter exactly 6 values.")
                continue

            X = np.array(legs, dtype=np.float32).reshape(1,-1)
            Xn = x_scaler.transform(X)
            Xn_t = torch.tensor(Xn, dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                pred_n = model(Xn_t).cpu().numpy()

            pred = y_scaler.inverse_transform(pred_n)[0]

            print("\nPredicted Pose:")
            print(f"X      : {pred[0]:.6f}")
            print(f"Y      : {pred[1]:.6f}")
            print(f"Z      : {pred[2]:.6f}")
            print(f"Thetax : {pred[3]:.6f}")
            print(f"Thetay : {pred[4]:.6f}")
            print(f"Thetaz : {pred[5]:.6f}\n")

        except Exception as e:
            print("Invalid input:", e)

# =========================
# RUN MANUAL MODE
# =========================
manual_fk_predict()