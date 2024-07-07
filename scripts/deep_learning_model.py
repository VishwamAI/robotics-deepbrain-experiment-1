import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os

def create_deep_learning_model(input_shape):
    """
    Create a deep learning model for EEG data interpretation and hologram generation.

    Args:
    input_shape (tuple): Shape of the input data (timesteps, features). For example, (1, 19679, 64).

    Returns:
    tf.keras.Model: Compiled deep learning model that outputs 3 continuous values for hologram parameters.
    """
    model = Sequential()

    # Input layer
    model.add(tf.keras.Input(shape=input_shape))

    # Convolutional layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # LSTM layers for capturing temporal dynamics
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dropout(0.5))

    # Fully connected layers for final prediction
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))  # Change output layer to produce 3 values with linear activation

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[])

    return model

def load_preprocessed_data(file_path):
    """
    Load preprocessed EEG data from a CSV file and normalize it.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    np.ndarray: Loaded and normalized data as a NumPy array.
    """
    df = pd.read_csv(file_path)
    data = df.values

    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Reshape the data to match the model's input shape
    data = data.reshape(1, 19679, 64)

    return data

def train_model(model, data, labels, epochs=10, batch_size=32):
    """
    Train the deep learning model.

    Args:
    model (tf.keras.Model): Compiled deep learning model.
    data (np.ndarray): Training data.
    labels (dict): Training labels with keys 'position', 'intensity', and 'shape'.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    tf.keras.callbacks.History: Training history.
    """
    # Combine the labels into a single array for training
    combined_labels = np.hstack((labels['position'], labels['intensity'].reshape(-1, 1), labels['shape'].reshape(-1, 1)))

    history = model.fit(data, combined_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def predict(model, data):
    """
    Make predictions using the trained model and post-process the output for hologram generation.

    Args:
    model (tf.keras.Model): Trained deep learning model.
    data (np.ndarray): Data to make predictions on.

    Returns:
    dict: Post-processed model predictions for hologram parameters.
    """
    predictions = model.predict(data)

    # Post-process the predictions to ensure they are suitable for hologram generation
    predictions = np.clip(predictions, 0, 1)

    # Separate the predictions into position, intensity, and shape
    position = predictions[:, :2]
    intensity = predictions[:, 2]
    shape = np.ones_like(intensity)  # Assuming shape is constant for simplicity

    return {
        "position": position,
        "intensity": intensity,
        "shape": shape
    }

if __name__ == "__main__":
    # Example usage
    input_shape = (1, 19679, 64)  # Example input shape (timesteps, features)
    model = create_deep_learning_model(input_shape)
    model.summary()

    # Load preprocessed data
    csv_dir = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/'
    sample_file = 'SUB_001_SIG_01.csv'
    file_path = os.path.join(csv_dir, sample_file)
    data = load_preprocessed_data(file_path)

    # Example labels (3 values for hologram generation)
    labels = {
        "position": np.random.rand(data.shape[0], 2),
        "intensity": np.random.rand(data.shape[0]),
        "shape": np.ones(data.shape[0])  # Assuming shape is constant for simplicity
    }

    # Train the model
    history = train_model(model, data, labels)

    # Make predictions
    predictions = predict(model, data)
    print("Predictions:", predictions)
