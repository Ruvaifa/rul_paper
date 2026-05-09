import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Set sequence length
sequence_length = 50

# Function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start in range(num_elements - seq_length):
        yield data_matrix[start:start + seq_length]

# Define file paths and column names
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f's_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# Read data
train = pd.read_csv('train_FD002.txt', sep='\s+', header=None, names=col_names)
test = pd.read_csv('test_FD002.txt', sep='\s+', header=None, names=col_names)
y_test = pd.read_csv('RUL_FD002.txt', sep='\s+', header=None, names=['RUL'])

# Function to add Remaining Useful Life (RUL)
def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    result_frame["RUL"] = result_frame["max_cycle"] - result_frame["time_cycles"]
    return result_frame.drop("max_cycle", axis=1)

train = add_remaining_useful_life(train)

# Add operating condition
def add_operating_condition(df):
    df_op_cond = df.copy()
    df_op_cond['setting_1'] = df_op_cond['setting_1'].round()
    df_op_cond['setting_2'] = df_op_cond['setting_2'].round(decimals=2)
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                            df_op_cond['setting_2'].astype(str) + '_' + \
                            df_op_cond['setting_3'].astype(str)
    return df_op_cond

X_train_condition = add_operating_condition(train)
X_test_condition = add_operating_condition(test)

# Function to scale features
def condition_scaler(df_train, df_test, sensor_names):
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test

# useful_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
useful_sensors = sensor_names
X_train_condition_scaled, X_test_condition_scaled = condition_scaler(X_train_condition, X_test_condition, useful_sensors)

seq_array = np.concatenate(list((list(gen_sequence(X_train_condition_scaled[X_train_condition_scaled['unit_nr'] == id], sequence_length, useful_sensors))
           for id in X_train_condition_scaled['unit_nr'].unique()))).astype(np.float32)
label_array = np.concatenate([X_train_condition_scaled[X_train_condition_scaled['unit_nr'] == id]['RUL'].values[sequence_length:] for id in X_train_condition_scaled['unit_nr'].unique()]).astype(np.float32)

# Convert to PyTorch tensors
seq_tensor = torch.tensor(seq_array).to(device)
label_tensor = torch.tensor(label_array).to(device)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()  # No activation needed in the final layer

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Only take the output from the last time step
        x = self.fc(x[:, -1, :])
        x = self.activation(x)  # Apply activation if necessary (identity in this case)
        return x

# Initialize model, loss function, and optimizer
model = LSTMModel(seq_array.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

# Initialize lists to store the loss values
train_loss_history = []
val_loss_history = []
rmse_history = []

num_epochs = 100
batch_size = 512
n_folds = 10

# 5-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(seq_tensor)):
    print(f'Fold {fold + 1}/{kf.n_splits}')
    
    seq_train, seq_val = seq_tensor[train_idx], seq_tensor[val_idx]
    label_train, label_val = label_tensor[train_idx], label_tensor[val_idx]
    
    # Reset the model for each fold
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    # Store validation loss per epoch
    val_loss_per_epoch = []
    train_loss_per_epoch = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_train_loss = 0

        for i in range(0, len(seq_train), batch_size):
            inputs = seq_train[i:i + batch_size]
            targets = label_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))  # Reshape targets
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / (len(seq_train) // batch_size)
        train_loss_per_epoch.append(avg_train_loss)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(seq_val)
            val_loss = criterion(val_outputs, label_val.view(-1, 1)).item()
            val_loss_per_epoch.append(val_loss)
            # if final epoch
            if epoch == num_epochs-1:
                predictions = model(seq_tensor)
                rmse_per_eppoch = torch.sqrt(criterion(predictions, label_tensor.view(-1, 1))).item()
                rmse_history.append(rmse_per_eppoch)
                print(f'RMSE: {rmse_per_eppoch}')
                print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {val_loss}')


    # Store all validation losses for this fold
    val_loss_history.append(val_loss_per_epoch)
    train_loss_history.append(train_loss_per_epoch)

def plot_loss(train_loss_history, val_loss_history):
    num_folds = len(train_loss_history)
    # Create subplots
    fig, axs = plt.subplots(num_folds, 1, figsize=(13, 5 * num_folds))
    for i in range(num_folds):
        axs[i].plot(train_loss_history[i], label='Training Loss')
        axs[i].plot(val_loss_history[i], label='Validation Loss')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
        axs[i].set_title(f'Training and Validation Loss (Fold {i + 1})')
    plt.tight_layout()
    plt.show()

# Plot the loss history after training
# plot_loss(train_loss_history, val_loss_history)

# Print rmse vs fold in a table
print('RMSE per fold:')
for i, rmse in enumerate(rmse_history):
    print(f'Fold {i + 1}: {round(rmse, 3)}')

# Print average RMSE
print(f'Average RMSE: {np.mean(rmse_history)}')
