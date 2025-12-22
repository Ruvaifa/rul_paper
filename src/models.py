import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)

# Check for MPS or GPU device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


class PenalizedMSELoss(nn.Module):
    def __init__(self, penalty_weight=0.1):
        """
        Custom loss function combining MSE with a penalty term.
        :param penalty_weight: Scaling factor for the penalty term.
        """
        super(PenalizedMSELoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        mse_loss = F.mse_loss(predictions, targets, reduction='mean')
        penalty = self.penalty_weight * torch.mean(torch.abs(predictions - targets)**3)
        return mse_loss + penalty


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


class StochasticLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, noise_std=0.1):
        super(StochasticLSTMCell, self).__init__(input_size, hidden_size)
        self.noise_std = noise_std  # Standard deviation of the noise

    def forward(self, input, hidden):
        # Standard LSTM cell computation
        h_next, c_next = super(StochasticLSTMCell, self).forward(input, hidden)

        # Add Gaussian noise to hidden state during training
        if self.training:
            noise = torch.randn_like(h_next) * self.noise_std
            h_next = h_next + noise
        return h_next, c_next


class MultiSensorLSTMModel(nn.Module):
    def __init__(self, input_size, num_features=21):
        super(MultiSensorLSTMModel, self).__init__()
        self.num_x_lstm2 = 50
        self.num_features = num_features
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention mechanism
        self.feature_fc = nn.Linear(self.num_features, self.num_features)  # For feature weighting
        self.time_attention_fc = nn.Linear(50, 1)  # For time step weighting
        
        # Pearson correlation calculation
        self.correlation_fc = nn.Linear(self.num_features, self.num_features) 
        
        # Fully connected layer for output
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()  # No activation needed in the final layer

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        
        # Step 1: Feature-level attention (multi-sensor weighting)
        feature_weights = F.softmax(self.feature_fc(x), dim=-1)  # Shape: (batch_size, seq_len, num_features)
        x_weighted = feature_weights * x  # Weighted feature matrix
        
        # Step 2: First LSTM layer
        x_lstm1, _ = self.lstm1(x_weighted)
        x_lstm1 = self.dropout1(x_lstm1)
        
        # Step 3: Second LSTM layer
        x_lstm2, _ = self.lstm2(x_lstm1)
        x_lstm2 = self.dropout2(x_lstm2)
        
        # Step 4: Pearson correlation-based attention
        # Compute correlation weights (simplified as another learned transformation)
        self.num_x_lstm2 = x_lstm2.size(1)
        correlation_weights = F.softmax(self.correlation_fc(x_weighted.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        # Project correlation_weights to [batch_size, seq_len=50]
        # Dynamically create the projection layer
        projection_fc = nn.Linear(self.num_features, self.num_x_lstm2).to(x.device)  # Ensure the layer is on the same device as the input
        correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, seq_len)
        correlation_weights = F.softmax(correlation_weights, dim=-1)
        # Expand correlation_weights to match x_lstm2
        correlation_weights = correlation_weights.unsqueeze(-1).expand(-1, -1, x_lstm2.size(-1))  # Shape: (batch_size, seq_len, hidden_size)
        x_corr_weighted = correlation_weights * x_lstm2  # Apply correlation weights
        
        # Step 5: Time attention mechanism
        # time_weights = F.softmax(self.time_attention_fc(x_corr_weighted), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x_corr_weighted, dim=1)  # Weighted sum across time steps
        attended_output = x_corr_weighted  # FOR WITHOUT ATTENTION 2048, 50, 50

        # Step 6: Final prediction
        output = self.fc(attended_output[:, -1, :]) #attended_output (with attention) or attended_output[:, -1, :] (without attention) # Only take the output from the last time step
        output = self.activation(output) # 2048, 1
        return output

class StochasticLSTMModel(nn.Module):
    def __init__(self, input_size, num_features=21, noise_std=0.1):
        super(StochasticLSTMModel, self).__init__()
        self.num_features = num_features
        
        # Stochastic LSTM layers
        self.stochastic_lstm1 = nn.LSTM(input_size, 100, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.stochastic_lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Replace LSTM cells with Stochastic LSTM cells
        self.stochastic_lstm1.cell = StochasticLSTMCell(input_size, 100, noise_std=noise_std)
        self.stochastic_lstm2.cell = StochasticLSTMCell(100, 50, noise_std=noise_std)
        
        # Attention mechanism
        self.feature_fc = nn.Linear(self.num_features, self.num_features)  # For feature weighting
        self.time_attention_fc = nn.Linear(50, 1)  # For time step weighting
        
        # Pearson correlation calculation
        self.correlation_fc = nn.Linear(self.num_features, self.num_features) 
        
        # Fully connected layer for output
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()  # No activation needed in the final layer

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        
        # Step 1: Feature-level attention (multi-sensor weighting)
        feature_weights = F.softmax(self.feature_fc(x), dim=-1)  # Shape: (batch_size, seq_len, num_features)
        x_weighted = feature_weights * x  # Weighted feature matrix
        
        # Step 2: First Stochastic LSTM layer
        x_lstm1, _ = self.stochastic_lstm1(x_weighted)
        x_lstm1 = self.dropout1(x_lstm1)
        
        # Step 3: Second Stochastic LSTM layer
        x_lstm2, _ = self.stochastic_lstm2(x_lstm1)
        x_lstm2 = self.dropout2(x_lstm2)
        
        # Step 4: Pearson correlation-based attention
        # Compute correlation weights (simplified as another learned transformation)
        self.num_x_lstm2 = x_lstm2.size(1)
        correlation_weights = F.softmax(self.correlation_fc(x_weighted.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        # Project correlation_weights to [batch_size, seq_len=50]
        # Dynamically create the projection layer
        projection_fc = nn.Linear(self.num_features, self.num_x_lstm2).to(x.device)  # Ensure the layer is on the same device as the input
        correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, seq_len)
        correlation_weights = F.softmax(correlation_weights, dim=-1)
        # Expand correlation_weights to match x_lstm2
        correlation_weights = correlation_weights.unsqueeze(-1).expand(-1, -1, x_lstm2.size(-1))  # Shape: (batch_size, seq_len, hidden_size)
        x_corr_weighted = correlation_weights * x_lstm2  # Apply correlation weights
        
        # Step 5: Time attention mechanism
        # time_weights = F.softmax(self.time_attention_fc(x_corr_weighted), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x_corr_weighted, dim=1)  # Weighted sum across time steps
        attended_output = x_corr_weighted  # FOR WITHOUT ATTENTION

        # Step 6: Final prediction
        output = self.fc(attended_output[:, -1, :]) #attended_output (with attention) or attended_output[:, -1, :] (without attention) # Only take the output from the last time step
        output = self.activation(output)
        return output
    

class StochasticMultiAttentionLSTMModel(nn.Module):
    def __init__(self, input_size, num_features=21, noise_std=0.1):
        super(StochasticMultiAttentionLSTMModel, self).__init__()
        self.num_features = num_features

        # Stochastic LSTM layers
        self.stochastic_lstm1 = nn.LSTM(input_size, 100, batch_first=True, dropout=0.2)
        self.stochastic_lstm2 = nn.LSTM(100, 50, batch_first=True, dropout=0.2)

        # Attention mechanisms
        self.feature_fc = nn.Linear(self.num_features, self.num_features)  # For feature weighting
        self.feature_scale = nn.Parameter(torch.ones(1))  # Learnable scaling factor for feature attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=50, num_heads=5, batch_first=True)  # Time attention

        # Nonlinear projection for correlation weights
        self.correlation_fc = nn.Sequential(
            nn.Linear(self.num_features, 50),
            nn.ReLU(),  # Nonlinear activation
            nn.Linear(50, 50)
        )

        # Fully connected layer for output
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()  # No activation needed in the final layer

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # Step 1: Feature-level attention (multi-sensor weighting)
        feature_weights = F.softmax(self.feature_scale * self.feature_fc(x), dim=-1)  # Shape: (batch_size, seq_len, num_features)
        x_weighted = feature_weights * x  # Weighted feature matrix

        # Step 2: First Stochastic LSTM layer
        x_lstm1, _ = self.stochastic_lstm1(x_weighted)

        # Step 3: Second Stochastic LSTM layer
        x_lstm2, _ = self.stochastic_lstm2(x_lstm1)

        # Step 4: Pearson correlation-based attention
        correlation_weights = F.softmax(self.correlation_fc(x_weighted.mean(dim=1)), dim=-1)  # Shape: (batch_size, hidden_size)
        correlation_weights = correlation_weights.unsqueeze(1).expand(-1, -1, x_lstm2.size(-1))  # Shape: (batch_size, seq_len, hidden_size)
        x_corr_weighted = correlation_weights * x_lstm2  # Apply correlation weights

        # Step 5: Multi-Head Time Attention
        attended_output, _ = self.multihead_attention(x_corr_weighted, x_corr_weighted, x_corr_weighted)  # Multi-head attention output

        # Step 6: Final prediction
        output = self.fc(attended_output[:, -1, :]) #attended_output (with attention) or attended_output[:, -1, :] # Use the output from the last time step
        output = self.activation(output) #2048, 1

        return output


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_features=21):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(num_features, 100)  # Project input to model dimension
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=100, nhead=4, dim_feedforward=200, dropout=0.2),
            num_layers=3
        )
        # Pearson correlation-based attention
        self.correlation_fc = nn.Linear(num_features, num_features)
        self.projection_fc = nn.Linear(num_features, 50)
        
        # Multi-head time attention
        self.time_attention_fc = nn.MultiheadAttention(embed_dim=100, num_heads=4, dropout=0.1)
        # self.time_attention_fc = nn.Linear(100, 1)  # For time step weighting
        
        self.fc = nn.Linear(100, 1)  # Final fully connected layer
        self.activation = nn.Identity()

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # Step 1: Pearson Correlation Attention
        correlation_weights = F.softmax(self.correlation_fc(x.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        projection_fc = nn.Linear(num_features, seq_len).to(x.device)  
        correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, 1, seq_len)
        correlation_weights = F.softmax(correlation_weights, dim=-1).unsqueeze(-1).expand(-1, -1 ,x.shape[2])  # Shape: (batch_size, seq_len, seq_len)
        x_corr_weighted = correlation_weights * x  # Apply correlation weights
        
        # Step 2: Transformer Encoder
        x = self.embedding(x_corr_weighted)  # Shape: (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # Transformer requires (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        
        # Step 3: Multi-Head Time Attention
        x = x.permute(1, 0, 2)  # MultiheadAttention expects (seq_len, batch_size, d_model)
        x, _ = self.time_attention_fc(x, x, x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        # time_weights = F.softmax(self.time_attention_fc(x), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x, dim=1)  # Weighted sum across time steps

        # Step 4: Final prediction (using last time step)
        output = self.fc(x[:, -1, :]) #attended_output for single attention or x[:, -1, :] for multihead Attention
        output = self.activation(output)
        return output



class HybridTransformerLSTMModel(nn.Module):
    def __init__(self, input_size, num_features=21):
        super(HybridTransformerLSTMModel, self).__init__()
        self.correlation_fc = nn.Linear(num_features, num_features)
        self.projection_fc = nn.Linear(num_features, 50)
        
        self.embedding = nn.Linear(num_features, 100)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=100, nhead=4, dim_feedforward=200, dropout=0.2),
            num_layers=2
        )
        self.lstm = nn.LSTM(100, 50, batch_first=True)
        
        # Multi-head time attention
        self.time_attention_fc = nn.MultiheadAttention(embed_dim=50, num_heads=2, dropout=0.1)
        # self.time_attention_fc = nn.Linear(50, 1)  # For time step weighting
        
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # Step 1: Pearson Correlation Attention
        correlation_weights = F.softmax(self.correlation_fc(x.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        projection_fc = nn.Linear(num_features, seq_len).to(x.device)  
        correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, 1, seq_len)
        correlation_weights = F.softmax(correlation_weights, dim=-1).unsqueeze(-1).expand(-1, -1 ,x.shape[2])  # Shape: (batch_size, seq_len, seq_len)
        x_corr_weighted = correlation_weights * x  # Apply correlation weights
        
        # Step 2: Transformer Encoder
        x = self.embedding(x_corr_weighted)  # Shape: (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # Transformer requires (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        
        # Step 3: LSTM Layer
        x, _ = self.lstm(x)
        
        # Step 3: Multi-Head Time Attention
        x = x.permute(1, 0, 2)  # MultiheadAttention expects (seq_len, batch_size, d_model)
        x, _ = self.time_attention_fc(x, x, x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        # time_weights = F.softmax(self.time_attention_fc(x), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x, dim=1)  # Weighted sum across time steps

        # Step 4: Final prediction (using last time step)
        output = self.fc(x[:, -1, :]) #attended_output for single attention or x[:, -1, :] for multihead Attention
        output = self.activation(output)
        return output


class CNNLSTMTransformerHybridModel(nn.Module):
    def __init__(self, input_size, num_features=21):
        super(CNNLSTMTransformerHybridModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=3, padding=1)  # Local feature extraction
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)  # Further abstraction
        
        # LSTM layers
        self.lstm = nn.LSTM(32, 50, batch_first=True)
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=50, nhead=2, dim_feedforward=100, dropout=0.2),
            num_layers=2
        )
        
        # Pearson correlation-based attention
        self.correlation_fc = nn.Linear(num_features, num_features)  # Feature weighting
        self.projection_fc = nn.Linear(num_features, 50)  # Match sequence length
        
        # Multi-Head Time Attention
        self.time_attention_fc = nn.MultiheadAttention(embed_dim=50, num_heads=2, dropout=0.1)
        # self.time_attention_fc = nn.Linear(50, 1)  # For time step weighting
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # Step 1: Pearson Correlation Attention
        correlation_weights = F.softmax(self.correlation_fc(x.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        projection_fc = nn.Linear(num_features, seq_len).to(x.device)  
        correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, 1, seq_len)
        correlation_weights = F.softmax(correlation_weights, dim=-1).unsqueeze(-1).expand(-1, -1 ,x.shape[2])  # Shape: (batch_size, seq_len, seq_len)
        x_corr_weighted = correlation_weights * x  # Apply correlation weights

        # Step 2: CNN layers
        x = x_corr_weighted.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_len) for CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_features)

        # Step 3: LSTM layers
        x, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size=50)

        # Step 4: Transformer Encoder
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, hidden_size)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_size)

        # Step 3: Multi-Head Time Attention
        x = x.permute(1, 0, 2)  # MultiheadAttention expects (seq_len, batch_size, d_model)
        x, _ = self.time_attention_fc(x, x, x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, d_model)
        # time_weights = F.softmax(self.time_attention_fc(x), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x, dim=1)  # Weighted sum across time steps

        # Step 4: Final prediction (using last time step)
        output = self.fc(x[:, -1, :]) #attended_output for single attention or x[:, -1, :] for multihead Attention
        output = self.activation(output)
        return output


class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, num_features=21):
        super(CNNLSTMModel, self).__init__()
        self.num_features = num_features

        self.feature_fc = nn.Linear(self.num_features, self.num_features)  # For feature weighting

        # CNN layers
        self.conv1 = nn.Conv1d(50, 64, kernel_size=3, padding=1)  # Local feature extraction
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # Further abstraction
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # LSTM layer
        self.lstm_1 = nn.LSTM(self.num_features, 150, batch_first=True)
        self.lstm_2 = nn.LSTM(150, 300, batch_first=True)
        self.lstm_3 = nn.LSTM(300, 150, batch_first=True)
        self.lstm_4 = nn.LSTM(150, 50, batch_first=True)

        self.lstm_5 = nn.LSTM(32, 150, batch_first=True)
        self.lstm_6 = nn.LSTM(150, 350, batch_first=True)
        self.lstm_7 = nn.LSTM(350, 150, batch_first=True)
        self.lstm_8 = nn.LSTM(150, 50, batch_first=True)

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.2)
        self.dropout_4 = nn.Dropout(0.2)
        
        self.dropout_5 = nn.Dropout(0.2)
        self.dropout_6 = nn.Dropout(0.2)
        self.dropout_7 = nn.Dropout(0.2)
        self.dropout_8 = nn.Dropout(0.2)
        
        # Pearson correlation-based attention
        self.correlation_fc = nn.Linear(num_features, num_features)  # Feature weighting
        self.projection_fc = nn.Linear(num_features, 50)  # Project features to match sequence length
        
        # Multi-Head Time Attention
        self.time_attention_fc = nn.MultiheadAttention(embed_dim=50, num_heads=2, dropout=0.1)
        # self.time_attention_fc = nn.Linear(50, 1)  # For time step weighting
        
        # Fully connected layer for prediction
        self.fc = nn.Linear(50, 1)
        self.activation = nn.Identity()

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        # # Step 1: Feature-level attention (multi-sensor weighting)
        feature_weights = F.softmax(self.feature_fc(x), dim=-1)  # Shape: (batch_size, seq_len, num_features)
        x_weighted = feature_weights * x  # Weighted feature matrix
        
        # Step 2: First LSTM layer
        x_lstm1, _ = self.lstm_1(x)
        x_lstm1 = self.dropout_1(x_lstm1)
        
        # Step 3: Second LSTM layer
        x_lstm2, _ = self.lstm_2(x_lstm1)
        x_lstm2 = self.dropout_2(x_lstm2)

        x_lstm3, _ = self.lstm_3(x_lstm2)
        x_lstm3 = self.dropout_3(x_lstm3)

        x_lstm4, _ = self.lstm_4(x_lstm3)
        x_lstm4 = self.dropout_4(x_lstm4)
        
        # # Step 4: Pearson correlation-based attention
        # # Compute correlation weights (simplified as another learned transformation)
        # self.num_x_lstm2 = x_lstm2.size(1)
        # correlation_weights = F.softmax(self.correlation_fc(x.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        # # Project correlation_weights to [batch_size, seq_len=50]
        # # Dynamically create the projection layer
        # projection_fc = nn.Linear(num_features, self.num_x_lstm2).to(x.device)  # Ensure the layer is on the same device as the input
        # correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, seq_len)
        # correlation_weights = F.softmax(correlation_weights, dim=-1)
        # # Expand correlation_weights to match x_lstm2
        # correlation_weights = correlation_weights.unsqueeze(-1).expand(-1, -1, x_lstm2.size(-1))  # Shape: (batch_size, seq_len, hidden_size)
        # x_corr_weighted = correlation_weights * x_lstm2  # Apply correlation weights
        
        # x_corr_weighted = x_lstm4

        # Step 5: CNN layers
        x = x_lstm4.permute(0, 2, 1)  # Change to (batch_size, num_features, seq_len) for CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, num_features)

        x, _ = self.lstm_5(x)
        x = self.dropout_5(x)

        x, _ = self.lstm_6(x)
        x = self.dropout_6(x)

        x, _ = self.lstm_7(x)
        x = self.dropout_7(x)

        x, _ = self.lstm_8(x)
        x = self.dropout_8(x)

        # # Step 4: Pearson correlation-based attention
        # # Compute correlation weights (simplified as another learned transformation)
        # self.num_x_lstm2 = x.size(1)
        # correlation_weights = F.softmax(self.correlation_fc(x_weighted.mean(dim=1)), dim=-1)  # Shape: (batch_size, num_features)
        # # Project correlation_weights to [batch_size, seq_len=50]
        # # Dynamically create the projection layer
        # projection_fc = nn.Linear(num_features, self.num_x_lstm2).to(x.device)  # Ensure the layer is on the same device as the input
        # correlation_weights = projection_fc(correlation_weights)  # Shape: (batch_size, seq_len)
        # correlation_weights = F.softmax(correlation_weights, dim=-1)
        # # Expand correlation_weights to match x_lstm2
        # correlation_weights = correlation_weights.unsqueeze(-1).expand(-1, -1, x_lstm2.size(-1))  # Shape: (batch_size, seq_len, hidden_size)
        # x_corr_weighted = correlation_weights * x_lstm2  # Apply correlation weights

        # x_corr_weighted = x_lstm4
        
        # Step 6: Time attention mechanism
        # time_weights = F.softmax(self.time_attention_fc(x), dim=1)  # Shape: (batch_size, seq_len, 1)
        # attended_output = torch.sum(time_weights * x, dim=1)  # Weighted sum across time steps
        attended_output = x  # FOR WITHOUT ATTENTION 2048, 50, 50

        # Step 7: Final prediction
        output = self.fc(attended_output[:, -1, :]) #attended_output (with attention) or attended_output[:, -1, :] (without attention) # Only take the output from the last time step
        output = self.activation(output) # 2048, 1
        return output