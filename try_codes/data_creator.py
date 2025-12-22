import pandas as pd
import numpy as np

# File paths (update with your actual file locations)
train_file = "train_FD001.txt"
test_file = "test_FD001.txt"
rul_file = "RUL_FD001.txt"

# Define column names for the CMAPSS dataset
columns = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
          [f"sensor_{i}" for i in range(1, 22)]

# Load data
train_data = pd.read_csv(train_file, sep=r'\s+', header=None, names=columns, engine='python').dropna(axis=1)
test_data = pd.read_csv(test_file, sep=r'\s+', header=None, names=columns, engine='python').dropna(axis=1)
rul_data = pd.read_csv(rul_file, header=None, names=["RUL"])

print("Train Data\n", train_data)
print("Test Data\n", test_data)
print("RUL Data\n", rul_data)

# Compute RUL for training data
max_cycles = train_data.groupby("unit_number")["time_in_cycles"].max()
train_data = train_data.merge(max_cycles.rename("max_cycles"), on="unit_number")
train_data["RUL"] = train_data["max_cycles"] - train_data["time_in_cycles"]
train_data = train_data.drop(columns=["max_cycles"])

print("Train Data\n", train_data)

# Add RUL to test data
last_cycles = test_data.groupby("unit_number")["time_in_cycles"].max()
test_data = test_data.merge(last_cycles.rename("last_cycle"), on="unit_number")
# test_data = test_data.merge(rul_data, left_on="unit_number", right_index=True)
test_data['unit_number'] = test_data['unit_number'].astype(str)  # or 'int64' if preferred
rul_data.index = rul_data.index.astype(str)  # assuming unit_number is in the index of rul_data
# Perform the merge
test_data = test_data.merge(rul_data, left_on="unit_number", right_index=True)
test_data["RUL"] = test_data["RUL"] + test_data["last_cycle"] - test_data["time_in_cycles"]
test_data = test_data.drop(columns=["last_cycle"])

print("Test Data\n", test_data)

# Normalize sensor data
sensor_columns = [col for col in train_data.columns if "sensor_" in col]
mean_std = train_data[sensor_columns].agg(["mean", "std"]).transpose()

for col in sensor_columns:
    mean, std = mean_std.loc[col]
    if std != 0:
        train_data[col] = (train_data[col] - mean) / std
        test_data[col] = (test_data[col] - mean) / std

print("Train Data\n", train_data)
print("Test Data\n", test_data)

# Save processed data
train_data.to_csv("processed_train_FD001.csv", index=False)
test_data.to_csv("processed_test_FD001.csv", index=False)

print("Data preprocessing complete. Files saved.")
