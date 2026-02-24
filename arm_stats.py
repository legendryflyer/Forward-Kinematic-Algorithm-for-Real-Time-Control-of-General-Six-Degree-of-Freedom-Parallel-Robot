import pandas as pd

# ===== CONFIG =====
FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\train_10k.xlsx"  
OUT_FILE = r"C:\Users\tavis\OneDrive\Documents\BARC\work\arm_min_max.txt"

ARM_COLS = ["Leg1","Leg2","Leg3","Leg4","Leg5","Leg6"]

# ===== LOAD DATA =====
df = pd.read_excel(FILE)

# ===== COMPUTE + SAVE =====
with open(OUT_FILE, "w") as f:
    f.write("ARM LENGTH LIMITS (from dataset)\n")
    f.write("================================\n\n")

    for col in ARM_COLS:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()

        f.write(f"{col}:\n")
        f.write(f"  Min  = {min_val:.6f}\n")
        f.write(f"  Max  = {max_val:.6f}\n")
        f.write(f"  Mean = {mean_val:.6f}\n\n")

print(f"\nSaved arm limits to:\n{OUT_FILE}")