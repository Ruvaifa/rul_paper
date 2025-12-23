import numpy as np
import pandas as pd

from src.train_test import train_test
from src.utils import plot_loss

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)

#num_epochs = 2500
num_epochs = 300
#batch_size = 2048
batch_size = 64
sequence_length = 50
#folds = 10
folds = 1

# models = ["CNNLSTMTransformerHybridModel", "HybridTransformerLSTMModel", "TransformerModel", "CNNLSTMModel",\
#             "StochasticLSTMModel", "MultiSensorLSTMModel", "StochasticMultiAttentionLSTMModel", "LSTMModel"]
models = ["LSTMModel"]


results_df = pd.DataFrame()
results_df["Model"] = models

for i in range(3, 4):
    print(f"\n\n\n<<<<<<<<<<<<<<<==File FD00{i}==>>>>>>>>>>>>>>>")
    train_R2 = []
    train_RMSE = []
    test_R2 = []
    test_RMSE = []
    for model in models:
        print(f"\n\n===============Training {model}===============")
        loss_history_fold = []
        train_history_fold = []
        val_history_fold = []
        train_rmse_fold = []
        train_variance_fold = []
        test_rmse_fold = []
        test_variance_fold = []
        for fold in range(folds):
            print(f"\nFold {fold + 1}/{folds}")
            loss_history, train_history, val_history, train_rmse, train_variance, test_rmse, test_variance = \
                train_test(i, num_epochs, batch_size, sequence_length, model, f"AllSensorsData{i}")
            loss_history_fold.append(loss_history)
            train_history_fold.append(train_history)
            val_history_fold.append(val_history)
            train_rmse_fold.append(train_rmse)
            train_variance_fold.append(train_variance)
            test_rmse_fold.append(test_rmse)
            test_variance_fold.append(test_variance)
        plot_loss(np.mean(np.asarray(train_history_fold), axis=0), np.mean(np.asarray(val_history_fold), axis=0), f"{model}AllSensorsData_try{i}")
        train_R2.append(np.mean(train_variance_fold))
        train_RMSE.append(np.mean(train_rmse_fold))
        test_R2.append(np.mean(test_variance_fold))
        test_RMSE.append(np.mean(test_rmse_fold))
    results_df[f"Train R2 FD00{i}"] = train_R2
    results_df[f"Train RMSE FD00{i}"] = train_RMSE
    results_df[f"Test R2 FD00{i}"] = test_R2
    results_df[f"Test RMSE FD00{i}"] = test_RMSE

results_df.to_csv("outputs_try/results.csv", index=False)


# TODO Improve model

# BEST is 19, 15, 19 LSTMModelAllSensors
# BEST is 17, 17, 15 CNNLSTMModelAllSensors (L-L-L-L)
# BEST is 16, 15, 15 CNNLSTMModelAllSensors (L-L-L-L-C-C-C-C-L-L-L-L)
