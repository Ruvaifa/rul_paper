import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Activation
from tensorflow.keras import layers
from tensorflow import keras

# Load the processed data
train_data = pd.read_csv('processed_train_FD001.csv')
test_data = pd.read_csv('processed_test_FD001.csv')

# List of useful sensors
useful_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

train_data = train_data[['unit_number'] + useful_sensors + ['RUL']]
test_data = test_data[['unit_number'] + useful_sensors + ['RUL']]

# Define sequence length
sequence_length = 70  # Number of cycles to consider in each sequence

# Prepare training data
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length):
            sequence = unit_data.iloc[i:i + sequence_length, 1:-1].values  # Exclude unit, cycle, RUL
            label = unit_data.iloc[i + sequence_length - 1, -1]  # RUL value
            sequences.append(sequence)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# Get sequences and labels for training
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Normalize the data using MinMaxScaler for input features and target variable
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Normalize RUL labels using MinMaxScaler
label_scaler = MinMaxScaler()
y_train_scaled = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = label_scaler.transform(y_test.reshape(-1, 1)).flatten()

# LSTM expects inputs in 3D shape: (samples, timesteps, features)
print(f"Training Data Shape: {X_train_scaled.shape}, Labels Shape: {y_train_scaled.shape}")
print(f"Test Data Shape: {X_test_scaled.shape}, Labels Shape: {y_test_scaled.shape}")

# # Model Building
# model = Sequential([
#     LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, X_train_scaled.shape[2])),
#     Dropout(0.2),
#     LSTM(50, activation='relu'),
#     Dropout(0.2),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Function to create a Transformer-based model
def create_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    # Encoder (MultiHead Attention Layer)
    x = layers.MultiHeadAttention(num_heads=8, key_dim=32)(inputs, inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward Layer (Position-wise)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output Layer
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(1)(x)  # Predicting RUL value

    model = keras.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Example input shape: (timesteps, features)
input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])  # Adjust to your dataset
model = create_transformer_model(input_shape)
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=64, validation_data=(X_test_scaled, y_test_scaled), 
                    callbacks=[early_stopping])

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict and Inverse Transform
y_pred_scaled = model.predict(X_test_scaled)
y_pred_rescaled = label_scaler.inverse_transform(y_pred_scaled)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rescaled))

# Compare predictions and true RUL values
results = pd.DataFrame({'True RUL': y_test, 'Predicted RUL': y_pred_rescaled.flatten()})
print(results.head())

# Plotting RMSE over epochs and validation loss
plt.figure(figsize=(12, 6))

# RMSE plot
# train_rmse = np.sqrt(np.array(history.history['loss']))
# val_rmse = np.sqrt(np.array(history.history['val_loss']))
# plt.subplot(1, 2, 1)
# plt.plot(train_rmse, label='Train RMSE')
# plt.plot(val_rmse, label='Validation RMSE')
# plt.legend()
# plt.title("RMSE Over Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("RMSE")

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.tight_layout()
plt.show()

print(f"Final RMSE: {rmse}")
