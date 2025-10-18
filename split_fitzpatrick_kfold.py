import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# --- Configuration ---
csv_path = "fitzpatrick17k.csv"       # your input CSV file
output_dir = "cv_splits"              # where to save splits
n_splits = 5
seed = 42                             # for reproducibility
label_col = "label"                   # column to stratify by

# --- Load dataset ---
df = pd.read_csv(csv_path)

# Make sure no NaN in label
df = df.dropna(subset=[label_col]).reset_index(drop=True)

# --- Prepare output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Stratified 5-fold split ---
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[label_col]), 1):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Save each fold’s CSV
    train_csv = os.path.join(output_dir, f"train_fold{fold}.csv")
    test_csv = os.path.join(output_dir, f"test_fold{fold}.csv")
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Fold {fold}: {len(train_df)} train / {len(test_df)} test")
