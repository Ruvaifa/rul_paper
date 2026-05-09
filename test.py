import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# Update this path
DATA_DIR = r"D:\Python codes\RUL\Task 7 - Copy\data"
# Example: DATA_DIR = r"D:\CMAPSS"

files = {
    "FD001": "train_FD001.txt",
    "FD002": "train_FD002.txt",
    "FD003": "train_FD003.txt",
    "FD004": "train_FD004.txt"
}

col_names = (
    ["engine_id", "cycle"] +
    [f"setting{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

# Create all figures first
for dataset, fname in files.items():
    path = os.path.join(DATA_DIR, fname)

    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = col_names[:df.shape[1]]

    lifetime_df = df.groupby("engine_id")["cycle"].max().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(lifetime_df["engine_id"], lifetime_df["cycle"], marker="o", linestyle="-")
    plt.title(f"Engine Lifetime vs Engine Number ({dataset})")
    plt.xlabel("Engine Number")
    plt.ylabel("Lifetime (Max Cycle)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

# Show everything at once
plt.show()
