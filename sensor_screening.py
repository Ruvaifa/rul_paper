import pandas as pd
import numpy as np
import os

# -----------------------------
# Metric functions
# -----------------------------

def sensor_variance(x):
    return x.var()

def sensor_monotonicity(x):
    diff = x.diff().dropna()
    if len(diff) == 0:
        return 0.0
    n_inc = (diff > 0).sum()
    n_dec = (diff < 0).sum()
    return abs(n_inc - n_dec) / len(diff)

def sensor_snr(x):
    diff = x.diff().dropna()
    if diff.std() == 0 or len(diff) == 0:
        return 0.0
    return diff.abs().mean() / (diff.std() + 1e-6)

# -----------------------------
# Main processing
# -----------------------------

def process_dataset(fd_id):
    print(f"Processing FD00{fd_id}")

    # Column names
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # Load training data only
    df = pd.read_csv(
        f"data/train_FD00{fd_id}.txt",
        sep=r"\s+",
        header=None,
        names=col_names
    )

    results = []

    # Loop over sensors
    for sensor in sensor_names:
        var_list = []
        mono_list = []
        snr_list = []

        # Compute per engine
        for unit in df['unit_nr'].unique():
            x = df[df['unit_nr'] == unit][sensor].reset_index(drop=True)

            var_list.append(sensor_variance(x))
            mono_list.append(sensor_monotonicity(x))
            snr_list.append(sensor_snr(x))

        results.append({
            "dataset": f"FD00{fd_id}",
            "sensor": sensor,
            "variance": np.mean(var_list),
            "monotonicity": np.mean(mono_list),
            "snr": np.mean(snr_list)
        })

    return results

# -----------------------------
# Run for all datasets
# -----------------------------

all_results = []

for fd in [1, 2, 3, 4]:
    all_results.extend(process_dataset(fd))

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Save to CSV
results_df.to_csv("sensor_screening_metrics.csv", index=False)

print("Saved sensor screening metrics to sensor_screening_metrics.csv")
