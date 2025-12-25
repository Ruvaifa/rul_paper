import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import os

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.models import (
    PenalizedMSELoss, 
    LSTMModel, 
    MultiSensorLSTMModel, 
    StochasticLSTMModel, 
    CNNLSTMModel,
    StochasticMultiAttentionLSTMModel, 
    TransformerModel, 
    HybridTransformerLSTMModel, 
    CNNLSTMTransformerHybridModel
)
from src.utils import evaluate
from src.data_preprocessor import prepare_data

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

# Define a learning rate scheduler
def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

def train_test(file_name, num_epochs, batch_size, sequence_length, model_name, msg=""):
    train_sequences, y_train, test_sequences, y_test, num_features = prepare_data(file_name, sequence_length)

    # Convert to PyTorch tensors
    # train_tensor = torch.tensor(train_sequences).to(device)
    # train_label = torch.tensor(y_train).to(device)

    train_tensor = torch.tensor(train_sequences)
    train_label = torch.tensor(y_train)


    # Initialize model, loss function, and optimizer
    if  model_name == "CNNLSTMTransformerHybridModel":
        model = CNNLSTMTransformerHybridModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "HybridTransformerLSTMModel":
        model = HybridTransformerLSTMModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "TransformerModel":
        model = TransformerModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "StochasticMultiAttentionLSTMModel":
        model = StochasticMultiAttentionLSTMModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "StochasticLSTMModel":
        model = StochasticLSTMModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "MultiSensorLSTMModel":
        model = MultiSensorLSTMModel(train_sequences.shape[2], num_features).to(device)
    elif model_name == "CNNLSTMModel":
        model = CNNLSTMModel(train_sequences.shape[2], num_features).to(device)
    else:
        model = LSTMModel(train_sequences.shape[2]).to(device)
    criterion = PenalizedMSELoss(penalty_weight=0.1) # PenalizedMSELoss(penalty_weight=0.1)  # nn.MSELoss()
    optimizer = optimizer = optim.AdamW(model.parameters(), lr=9.5e-4, weight_decay=0.01) #optim.RMSprop(model.parameters())
    #scheduler = get_scheduler(optimizer, num_warmup_steps=300, num_training_steps=batch_size*num_epochs)
    num_training_steps = (len(train_tensor) // batch_size) * num_epochs
    scheduler = get_scheduler(optimizer, num_warmup_steps=300, num_training_steps=num_training_steps)


    # Initialize a list to store the loss values
    loss_history = []
    train_history = []
    val_history = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    best_model_path = f"outputs/{model_name}_{msg}"
    last_model_path = ""

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0
        for i in range(0, len(train_tensor), batch_size):
            # inputs = train_tensor[i:i + batch_size]
            # targets = train_label[i:i + batch_size]
            inputs = train_tensor[i:i + batch_size].to(device)
            targets = train_label[i:i + batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))  # Reshape targets
            loss.backward()
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(train_tensor) // batch_size)
        loss_history.append(avg_loss)  # Store the average loss for this epoch

        # Validation step (val_set = test_set for now)
        model.eval()
        # with torch.no_grad():
        #     predictions = model(train_tensor)
        #     train_label_copy = train_label
        #     train_rmse,_ = evaluate(train_label_copy.detach().cpu().numpy(), predictions.detach().cpu().numpy(), f"Train_{file_name}", False)
        with torch.no_grad():
            train_preds = []
            train_true = []

            for i in range(0, len(train_tensor), batch_size):
                batch_x = train_tensor[i:i+batch_size].to(device)
                batch_y = train_label[i:i+batch_size]

                preds = model(batch_x).cpu()
                train_preds.append(preds)
                train_true.append(batch_y)
#################################################
            train_preds = torch.cat(train_preds).numpy()
            train_true = torch.cat(train_true).numpy()
###########################################################
            train_rmse, train_variance = evaluate(
                train_true, train_preds, f"Train_{file_name}", False
            )
            
            
            train_history.append(train_rmse)
            y_hat_test = []
            for test_item in test_sequences:
                test_tensor = torch.tensor(np.asarray([test_item]).astype(np.float32)).to(device)
                test_shape = test_tensor.shape
                if test_shape[0]==0 or test_shape[1]==0 or test_shape[2]==0:
                    continue
                y_hat_test.append(model(test_tensor))
            y_hat_test = torch.cat(y_hat_test).detach().cpu().numpy()
   
            val_rmse,_ = evaluate(y_test, y_hat_test, f"Test_{file_name}", False)
            
            val_history.append(val_rmse)

            if epoch == num_epochs - 1:
                last_model_path = f"{best_model_path}.pt"
                torch.save(model.state_dict(), last_model_path)
                
            # Early stopping check
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                patience_counter = 0  # Reset the counter if validation loss improves
                # Save the model
                if os.path.exists(last_model_path):
                    os.remove(last_model_path)
                last_model_path = f"{best_model_path}_epoch{epoch}.pt"
                torch.save(model.state_dict(), last_model_path)
            else:
                patience_counter += 1  # Increment the counter if validation loss does not improve
####################################################
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
#####################################################
    print (f"Train RMSE: {train_rmse}, Val RMSE: {val_rmse}")

    # Load the best model for final evaluation
    print(f"Loading the best model for final evaluation")
    model.load_state_dict(torch.load(last_model_path))

    # Evaluate the model
    model.eval()
    # with torch.no_grad():
    #     predictions = model(train_tensor)
    #     train_rmse, train_variance = evaluate(train_label.cpu().numpy(), predictions.cpu().numpy(), f"Train_{file_name}", False)
    with torch.no_grad():
        train_preds = []
        train_true = []

        for i in range(0, len(train_tensor), batch_size):
            batch_x = train_tensor[i:i+batch_size].to(device)
            batch_y = train_label[i:i+batch_size]

            preds = model(batch_x).cpu()
            train_preds.append(preds)
            train_true.append(batch_y)

        train_preds = torch.cat(train_preds).numpy()
        train_true = torch.cat(train_true).numpy()

        train_rmse, train_variance = evaluate(
            train_true, train_preds, f"Train_{file_name}", False
        )


        y_hat_test = []
        for test_item in test_sequences:
            test_tensor = torch.tensor(np.asarray([test_item]).astype(np.float32)).to(device)
            test_shape = test_tensor.shape
            if test_shape[0]==0 or test_shape[1]==0 or test_shape[2]==0:
                continue
            y_hat_test.append(model(test_tensor))
        y_hat_test = torch.cat(y_hat_test).cpu().numpy()

        print("y_test min/max:", y_test.min(), y_test.max())
        print("y_hat min/max:", y_hat_test.min(), y_hat_test.max())
        test_rmse, test_variance = evaluate(y_test, y_hat_test, f"Test_{file_name}", False)

    print(f"Train RMSE: {train_rmse}, Train R2:{train_variance}, Test RMSE: {test_rmse}, Test R2: {test_variance}")

    return loss_history, train_history, val_history, train_rmse, train_variance, test_rmse, test_variance

