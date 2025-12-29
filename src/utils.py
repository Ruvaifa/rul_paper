from sklearn.preprocessing import StandardScaler # type: ignore
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)

# Function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # yield data_matrix[:seq_length]
    for start in range(num_elements - seq_length):
        yield data_matrix[start:start + seq_length]


# Function to add Remaining Useful Life (RUL)
def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["time_cycles"]
    return result_frame.drop("max_cycle", axis=1)


# Function to scale features
# def condition_scaler(df_train, df_test, sensor_names):
#     scaler = StandardScaler()
#     for condition in df_train['op_cond'].unique():
#         scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
#         df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
#         df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
#     return df_train, df_test
def condition_scaler(train_df, test_df, sensors, file_id=None):
    """
    Regime-based normalization for FD002 & FD004.
    Global normalization for others.
    """

    # ===== FD002 & FD004 ONLY =====
    if file_id == 2 or file_id == 4:

        # 1. Learn regimes from TRAIN data
        kmeans = KMeans(n_clusters=6, random_state=42)
        train_df['regime'] = kmeans.fit_predict(
            train_df[['setting_1', 'setting_2', 'setting_3']]
        )

        # Assign regimes to test
        test_df['regime'] = kmeans.predict(
            test_df[['setting_1', 'setting_2', 'setting_3']]
        )

        # 2. Normalize PER REGIME
        for r in range(6):
            train_mask = train_df['regime'] == r
            test_mask = test_df['regime'] == r

            mean = train_df.loc[train_mask, sensors].mean()
            std = train_df.loc[train_mask, sensors].std()

            std[std == 0] = 1.0  # safety

            train_df.loc[train_mask, sensors] = (
                train_df.loc[train_mask, sensors] - mean
            ) / std

            test_df.loc[test_mask, sensors] = (
                test_df.loc[test_mask, sensors] - mean
            ) / std
            
        return train_df, test_df

    # ===== FD001 & FD003 (original behavior) =====
    else:
        # mean = train_df[sensors].mean()
        # std = train_df[sensors].std()
        # std[std == 0] = 1.0

        # train_df[sensors] = (train_df[sensors] - mean) / std
        # test_df[sensors] = (test_df[sensors] - mean) / std

        # return train_df, test_df
        scaler = StandardScaler()
        for condition in train_df['op_cond'].unique():
            scaler.fit(train_df.loc[train_df['op_cond'] == condition, sensors])
            train_df.loc[train_df['op_cond'] == condition, sensors] = scaler.transform(train_df.loc[train_df['op_cond'] == condition, sensors])
            test_df.loc[test_df['op_cond'] == condition, sensors] = scaler.transform(test_df.loc[test_df['op_cond'] == condition, sensors])
        return train_df, test_df

# Add operating condition
def add_operating_condition(df):
    df_op_cond = df.copy()
    df_op_cond['setting_1'] = df_op_cond['setting_1'].round()
    df_op_cond['setting_2'] = df_op_cond['setting_2'].round(decimals=2)
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)
    return df_op_cond

# Evaluate the predictions
def evaluate(y_true, y_hat, label='test', to_print=True):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    if to_print:
        print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
    return rmse, variance

# Plotting loss history
def plot_loss(train_loss, val_loss, msg):
    plt.figure(figsize=(13, 5))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(msg)
    plt.legend()
    # plt.show()
    plt.savefig(f"outputs/{msg}.png")