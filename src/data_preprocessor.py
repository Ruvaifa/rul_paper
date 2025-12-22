import pandas as pd
import numpy as np
try:
    from src.utils import gen_sequence, add_operating_condition, add_remaining_useful_life, condition_scaler
except ModuleNotFoundError:
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from src.utils import gen_sequence, add_operating_condition, add_remaining_useful_life, condition_scaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)

def prepare_data(file_name, sequence_length):
    MAX_WINDOWS_PER_ENGINE = 150
    

    # Define file paths and column names
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # Read data
    train = pd.read_csv(f'data/train_FD00{file_name}.txt', sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv(f'data/test_FD00{file_name}.txt', sep=r'\s+', header=None, names=col_names)
    y_test = pd.read_csv(f'data/RUL_FD00{file_name}.txt', sep=r'\s+', header=None, names=['RUL'])

    # Add RUL to training data
    train = add_remaining_useful_life(train)

    print("Dataset FD00", file_name)
    print("Train rows:", train.shape)
    print("Test rows:", test.shape)
    print("Number of engines:", train['unit_nr'].nunique())

    # Preprocess and add operating condition
    X_train_condition = add_operating_condition(train)
    X_test_condition = add_operating_condition(test)

    # Set useful tensors
    useful_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    # useful_sensors = sensor_names

    # Scale the array
    X_train_condition_scaled, X_test_condition_scaled = condition_scaler(X_train_condition, X_test_condition, useful_sensors)
#---------------------------TRAINING-----------------------------------
    # Create training sequences
    # train_sequences = np.concatenate(list((list(gen_sequence(X_train_condition_scaled[X_train_condition_scaled['unit_nr']==id], sequence_length, useful_sensors))
    #             for id in X_train_condition_scaled['unit_nr'].unique()))).astype(np.float32)
    ######################
    # train_sequences = []

    # for id in X_train_condition_scaled['unit_nr'].unique():
    #     unit_df = X_train_condition_scaled[
    #         X_train_condition_scaled['unit_nr'] == id
    #     ]

    #     sequences = list(gen_sequence(unit_df, sequence_length, useful_sensors))

    #     # Apply window capping
    #     if len(sequences) > MAX_WINDOWS_PER_ENGINE:
    #         sequences = sequences[-MAX_WINDOWS_PER_ENGINE:]  # keep last K windows

    #     train_sequences.extend(sequences)

    # train_sequences = np.asarray(train_sequences, dtype=np.float32)

    train_sequences = []
    train_labels = []

    for id in X_train_condition_scaled['unit_nr'].unique():
        unit_df = X_train_condition_scaled[
            X_train_condition_scaled['unit_nr'] == id
        ]

        # Generate sensor windows
        x_seq = list(gen_sequence(unit_df, sequence_length, useful_sensors))

        # Generate RUL windows
        y_seq = list(gen_sequence(unit_df, sequence_length, ['RUL']))

        # Take RUL from last timestep of each window
        y_seq = [y[-1][0] for y in y_seq]

        # Safety check
        assert len(x_seq) == len(y_seq)

        # Apply window capping (FAIRNESS FIX)
        if len(x_seq) > MAX_WINDOWS_PER_ENGINE:
            x_seq = x_seq[-MAX_WINDOWS_PER_ENGINE:]
            y_seq = y_seq[-MAX_WINDOWS_PER_ENGINE:]

        train_sequences.extend(x_seq)
        train_labels.extend(y_seq)

    train_sequences = np.asarray(train_sequences, dtype=np.float32)
    y_train = np.asarray(train_labels, dtype=np.float32)
#---------------------------TESTING-------------------------------------
    engine_window_counts = {}

    for id in X_train_condition_scaled['unit_nr'].unique():
        unit_len = len(X_train_condition_scaled[X_train_condition_scaled['unit_nr'] == id])
        windows = max(0, unit_len - sequence_length)
        engine_window_counts[id] = windows

    print("Training windows per engine:")
    print("Min:", min(engine_window_counts.values()))
    print("Max:", max(engine_window_counts.values()))
    print("Mean:", np.mean(list(engine_window_counts.values())))
# ------------------------------------------------------------------------------        
    # Create testing sequences
    # test_sequences = []
    # for id in X_test_condition_scaled['unit_nr'].unique():
    #     unit_data = X_test_condition_scaled[X_test_condition_scaled['unit_nr'] == id]
    #     sequence_length_for_unit_data = len(unit_data)-1
    #     sequences = list(gen_sequence(X_test_condition_scaled[X_test_condition_scaled['unit_nr'] == id], sequence_length_for_unit_data, useful_sensors))
    #     if len(sequences) > 0:
    #         test_sequences.extend(sequences)
    #     else:
    #         print(f"Skipping engine {id} due to insufficient time steps for sequence generation.")

    test_sequences = []

    for id in X_test_condition_scaled['unit_nr'].unique():
        unit_data = X_test_condition_scaled[
            X_test_condition_scaled['unit_nr'] == id
        ][useful_sensors].values  # shape: (num_cycles, num_features)

        num_cycles = unit_data.shape[0]

        if num_cycles >= sequence_length:
        # Take last `sequence_length` cycles
            seq = unit_data[-sequence_length:]
        else:
        # Left-pad with zeros
            pad_len = sequence_length - num_cycles
            pad = np.zeros((pad_len, unit_data.shape[1]))
            seq = np.vstack((pad, unit_data))

        test_sequences.append(seq.astype(np.float32))

    if len(test_sequences) != y_test.shape[0]:
        print(
            f"Aligning test sequences: "
            f"{len(test_sequences)} → {y_test.shape[0]}"
    )
    test_sequences = test_sequences[:y_test.shape[0]]
# ------------------------------------------------------------------------------
    # Extract training labels
    # y_train = np.concatenate([X_train_condition_scaled[X_train_condition_scaled['unit_nr'] == id]['RUL'].values[sequence_length:] for id in X_train_condition_scaled['unit_nr'].unique()]).astype(np.float32)
    # y_train = np.concatenate(list((list(gen_sequence(X_train_condition_scaled[X_train_condition_scaled['unit_nr']==id], sequence_length, ['RUL']))                 
    #             for id in X_train_condition_scaled['unit_nr'].unique()))).astype(np.float32)
    # y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])[:, -1].flatten()


#----------

    print("Train sequences shape:", train_sequences.shape)
    print("y_train shape:", y_train.shape)
    print("Test engines:", len(test_sequences))
    print("y_test shape:", y_test.shape)

    print("Unique test engines (from data):",X_test_condition_scaled['unit_nr'].nunique())
    # print("Test sequences created:",len(test_sequences))
    print("RUL labels:",y_test.shape[0])

    return train_sequences, y_train, test_sequences, y_test, len(useful_sensors)
