import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Check for MPS or GPU device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("gpu")
else:
    device = torch.device("cpu")
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
train = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=col_names)
test = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None, names=col_names)
y_test = pd.read_csv('data/RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

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

seq_array = np.concatenate(list((list(gen_sequence(X_train_condition_scaled[X_train_condition_scaled['unit_nr']==id], sequence_length, useful_sensors))
            for id in X_train_condition_scaled['unit_nr'].unique()))).astype(np.float32)
test_sequences = []
for id in X_test_condition_scaled['unit_nr'].unique():
    unit_data = X_test_condition_scaled[X_test_condition_scaled['unit_nr'] == id]
    sequence_length_for_unit_data = len(unit_data)-1
    sequences = list(gen_sequence(X_test_condition_scaled[X_test_condition_scaled['unit_nr'] == id], sequence_length_for_unit_data, useful_sensors))
    if len(sequences) > 0:
        test_sequences.extend(sequences)
    else:
        print(f"Skipping engine {id} due to insufficient time steps for sequence generation.")

xTestDFshape = X_test_condition_scaled.shape
xTrainDFshape = X_train_condition_scaled.shape

# test_array = np.concatenate(test_sequences).astype(np.float32)
# test_array = np.asarray(test_sequences).astype(np.float32)
# test_array = np.concatenate(list((list(gen_sequence(X_test_condition_scaled[X_test_condition_scaled['unit_nr']==id], sequence_length, useful_sensors))
#             for id in X_test_condition_scaled['unit_nr'].unique()))).astype(np.float32)
label_array = np.concatenate([X_train_condition_scaled[X_train_condition_scaled['unit_nr'] == id]['RUL'].values[sequence_length:] for id in X_train_condition_scaled['unit_nr'].unique()]).astype(np.float32)

# Convert to PyTorch tensors
seq_tensor = torch.tensor(seq_array).to(device)
# test_tensor = torch.tensor(test_sequences).to(device)
label_tensor = torch.tensor(label_array).to(device)

# # Define the LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size):
#         super(LSTMModel, self).__init__()
#         self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
#         self.dropout1 = nn.Dropout(0.2)
#         self.lstm2 = nn.LSTM(100, 50, batch_first=True)
#         self.dropout2 = nn.Dropout(0.2)
#         self.fc = nn.Linear(50, 1)
#         self.activation = nn.Identity()  # No activation needed in the final layer

#     def forward(self, x):
#         # First LSTM layer
#         x, _ = self.lstm1(x)
#         x = self.dropout1(x)
        
#         # Second LSTM layer
#         x, _ = self.lstm2(x)
#         x = self.dropout2(x)

#         # Only take the output from the last time step
#         x = self.fc(x[:, -1, :])
#         x = self.activation(x)  # Apply activation if necessary (identity in this case)
#         return x

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention layer
        self.attention = nn.Linear(50, 1)  # Single layer attention mechanism
        
        # Fully connected layer for final output
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()  # No activation needed in the final layer

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Apply attention on the second LSTM output
        attention_weights = F.softmax(self.attention(x), dim=1)  # Shape: (batch_size, sequence_length, 1)
        attended_output = torch.sum(attention_weights * x, dim=1)  # Weighted sum of LSTM outputs

        # Pass the attended output through the final fully connected layer
        x = self.fc(attended_output)
        x = self.activation(x)  # Apply activation if necessary (identity in this case)
        return x


# Initialize model, loss function, and optimizer
model = LSTMModel(seq_array.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

# Initialize a list to store the loss values
loss_history = []

# Train the model
num_epochs = 3
batch_size = 1024

for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    for i in range(0, len(seq_tensor), batch_size):
        inputs = seq_tensor[i:i + batch_size]
        targets = label_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))  # Reshape targets
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (len(seq_tensor) // batch_size)
    loss_history.append(avg_loss)  # Store the average loss for this epoch
    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')

def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))

# Evaluate the model
model.eval()
with torch.no_grad():
    # print (seq_tensor)
    predictions = model(seq_tensor)
    rmse = torch.sqrt(criterion(predictions, label_tensor.view(-1, 1))).item()
    # print(f'RMSE: {rmse}')
    evaluate(label_tensor.cpu().numpy(), predictions.cpu().numpy(), "Train")
    # print(test_tensor)
    y_hat_test = []
    for test_item in test_sequences:
        test_tensor = torch.tensor(np.asarray([test_item]).astype(np.float32)).to(device)
        test_shape = test_tensor.shape
        if test_shape[0]==0 or test_shape[1]==0 or test_shape[2]==0:
            continue
        y_hat_test.append(model(test_tensor))
    y_hat_test = torch.cat(y_hat_test).cpu().numpy()
    evaluate(y_test, y_hat_test, "Test")

# Plotting loss history
def plot_loss(loss_history):
    plt.figure(figsize=(13, 5))
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the loss history after training
# plot_loss(loss_history)