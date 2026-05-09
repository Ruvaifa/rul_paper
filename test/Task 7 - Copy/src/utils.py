from sklearn.preprocessing import StandardScaler # type: ignore
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
def condition_scaler(df_train, df_test, sensor_names):
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


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