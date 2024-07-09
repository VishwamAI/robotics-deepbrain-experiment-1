import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from data_preprocessing import load_csv_data_in_chunks

def state_transition_model(state_vectors, transition_matrix, process_noise_cov):
    """
    State transition model for the multicore BPF approach.

    Args:
    state_vectors (tf.Tensor): Current state vectors for all particles (num_particles, 3D coordinates).
    transition_matrix (tf.Tensor): State transition matrix.
    process_noise_cov (tf.Tensor): Process noise covariance matrix.

    Returns:
    tf.Tensor: Updated state vectors based on the state transition model.
    """
    updated_state_vectors = tf.linalg.matmul(tf.cast(state_vectors, tf.float64), transition_matrix, transpose_b=True)
    process_noise_cov = tf.cast(process_noise_cov, tf.float32)  # Ensure dtype consistency
    # Verify that process_noise_cov is positive-definite
    tf.debugging.assert_positive(tf.linalg.det(process_noise_cov), message="process_noise_cov must be positive-definite")
    process_noise = tf.random.normal(shape=(state_vectors.shape[0], state_vectors.shape[1]), mean=0.0, stddev=1.0)
    process_noise = tf.linalg.matmul(process_noise, tf.linalg.cholesky(process_noise_cov))  # Generate correlated noise
    updated_state_vectors = tf.cast(updated_state_vectors, tf.float32) + process_noise

    return updated_state_vectors

def eeg_measurement_model(state_vector, forward_matrix):
    """
    EEG measurement model for the multicore BPF approach.

    Args:
    state_vector (tf.Tensor): Current state vector (num_particles, 3D coordinates), reshaped to (3, num_particles).
    forward_matrix (tf.Tensor): Forward matrix for the head model.

    Returns:
    tf.Tensor: Predicted EEG measurements based on the state vector.
    """
    # Ensure state_vector has the correct shape for matrix multiplication
    state_vector = tf.reshape(state_vector, [-1, 3])
    # Transpose state_vector for correct matrix multiplication
    state_vector = tf.transpose(state_vector)
    # Cast forward_matrix to tf.float32 to match state_vector
    forward_matrix = tf.cast(forward_matrix, tf.float32)
    predicted_measurements = tf.linalg.matmul(forward_matrix, state_vector)
    return tf.cast(predicted_measurements, tf.float32)

def apply_spatial_filter(eeg_data, filter_matrices):
    """
    Apply a spatial filter to the EEG data based on beamforming principles.

    Args:
    eeg_data (tf.Tensor): Input EEG data (timesteps, channels).
    filter_matrices (tf.Tensor): Time-dependent spatial filter matrices (timesteps, channels, channels).

    Returns:
    tf.Tensor: Spatially filtered EEG data.
    """
    num_timesteps, num_channels = eeg_data.shape
    if filter_matrices.shape[0] != num_timesteps or filter_matrices.shape[1] != num_channels:
        raise ValueError("The shape of the filter matrices must match the shape of the EEG data.")

    def apply_filter(t):
        return tf.linalg.matmul(filter_matrices[t], eeg_data[t, :])

    filtered_data = tf.map_fn(apply_filter, tf.range(num_timesteps), dtype=tf.float32)

    return filtered_data

def generate_beamforming_matrix(eeg_data, forward_matrices, state_transition_model, transition_matrix, process_noise_cov):
    """
    Generate a spatial filter matrix based on beamforming principles.

    Args:
    eeg_data (tf.Tensor): Input EEG data (timesteps, channels).
    forward_matrices (tf.Tensor): Precomputed forward matrices for the head model.
    state_transition_model (function): State transition model function.
    transition_matrix (tf.Tensor): State transition matrix.
    process_noise_cov (tf.Tensor): Process noise covariance matrix.

    Returns:
    tf.Tensor: Time-dependent spatial filter matrix (timesteps, channels, channels).
    """
    num_timesteps, num_channels = eeg_data.shape
    if num_channels <= 0:
        raise ValueError("The number of channels must be greater than zero.")

    # Initialize the state vector for each time step
    state_vector = tf.zeros((num_timesteps, num_channels, 3))  # Assuming 3D coordinates for each channel

    # Apply the state transition model for each time step
    for t in range(num_timesteps):
        for i in range(num_channels):
            state_vector[t, i] = state_transition_model(state_vector[t, i], transition_matrix, process_noise_cov)

    # Compute the beamforming matrix using the forward matrices and state vector
    filter_matrices = tf.zeros((num_timesteps, num_channels, num_channels))
    for t in range(num_timesteps):
        for i in range(num_channels):
            filter_matrices[t, i, :] = tf.reduce_mean(forward_matrices[:, i, :], axis=0) * state_vector[t, i]

    return filter_matrices

class MulticoreBPFLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow layer implementing a multicore Bayesian Particle Filter (BPF).

    This layer applies a state transition model, computes particle weights based on EEG measurements,
    resamples particles, and returns the mean of the resampled state vector as a single value output.

    Args:
    num_particles (int): Number of particles for the particle filter.
    transition_matrix (np.ndarray): State transition matrix.
    process_noise_cov (np.ndarray): Process noise covariance matrix.
    forward_matrix (np.ndarray): Forward matrix for the head model.
    state_vector (np.ndarray): State vector for the particles, reshaped to (3, num_particles) in the call method.
    """
    def __init__(self, num_particles, transition_matrix, process_noise_cov, forward_matrix, **kwargs):
        super(MulticoreBPFLayer, self).__init__(**kwargs)
        self.num_particles = num_particles
        self.transition_matrix = transition_matrix
        self.process_noise_cov = process_noise_cov
        self.forward_matrix = forward_matrix

    def build(self, input_shape):
        self.state_vector = self.add_weight(shape=(self.num_particles, 3), initializer='random_normal', trainable=False, name='state_vector')
        self.particle_weights = self.add_weight(shape=(self.num_particles,), initializer='ones', trainable=False, name='particle_weights')

    @tf.function
    def call(self, inputs):
        # Apply the state transition model
        self.state_vector.assign(state_transition_model(self.state_vector, self.transition_matrix, self.process_noise_cov))

        # Reshape state_vector to match the expected shape for matrix multiplication
        reshaped_state_vector = tf.reshape(self.state_vector, [self.num_particles, 3])  # Reshape to (num_particles, 3)
        tf.print("Shape of reshaped_state_vector:", tf.shape(reshaped_state_vector))

        # Compute particle weights based on the EEG measurement model
        predicted_measurements = tf.cast(eeg_measurement_model(reshaped_state_vector, self.forward_matrix), tf.float32)
        tf.print("Shape of predicted_measurements:", tf.shape(predicted_measurements))

        # Print shapes for debugging
        tf.print("Shape of inputs:", tf.shape(inputs))
        tf.print("Inputs:", inputs, summarize=-1)
        tf.print("Shape of predicted_measurements:", tf.shape(predicted_measurements))
        tf.print("Predicted measurements:", predicted_measurements, summarize=-1)
        tf.print("Shape of forward_matrix:", tf.shape(self.forward_matrix))
        tf.print("Shape of state_vector:", tf.shape(self.state_vector))

        input_shape = tf.shape(inputs)
        predicted_shape = tf.shape(predicted_measurements)

        # Log the actual values of the shapes for debugging
        tf.print("Actual input shape:", input_shape)
        tf.print("Actual predicted shape:", predicted_shape)

        # Ensure the total number of elements in inputs matches predicted_measurements
        if tf.reduce_prod(input_shape[1:]) != tf.reduce_prod(predicted_shape):
            tf.print("Dimension mismatch: reshaped inputs shape", input_shape[1:], "does not match predicted_measurements shape", predicted_shape)
            raise ValueError(f"Dimension mismatch: reshaped inputs shape {input_shape[1:]} does not match predicted_measurements shape {predicted_shape}")

        # Reshape inputs to match the shape of predicted_measurements
        reshaped_inputs = tf.reshape(inputs, [input_shape[0], predicted_shape[0], predicted_shape[1]])
        tf.print("Shape of reshaped_inputs:", tf.shape(reshaped_inputs))

        self.particle_weights.assign(tf.reduce_sum(tf.square(reshaped_inputs - tf.transpose(predicted_measurements)), axis=-1))

        # Resample particles based on weights
        resampled_indices = tf.random.categorical(tf.math.log(tf.expand_dims(self.particle_weights, 0)), self.num_particles)
        resampled_state_vector = tf.gather(self.state_vector, tf.squeeze(resampled_indices, axis=0))

        # Return the mean of the resampled state vector as the single value output
        return tf.reduce_mean(resampled_state_vector, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 3)

def create_deep_learning_model(input_shape, transition_matrix, process_noise_cov, forward_matrix):
    """
    Create a deep learning model for EEG data interpretation and hologram generation.

    Args:
    input_shape (tuple): Shape of the input data (timesteps, features).
    transition_matrix (np.ndarray): State transition matrix.
    process_noise_cov (np.ndarray): Process noise covariance matrix.
    forward_matrix (np.ndarray): Forward matrix for the head model (num_particles, 3).

    Returns:
    tf.keras.Model: Compiled deep learning model that outputs a single continuous value per sample.

    Note:
    This function has been updated to better reflect the specific requirements
    of the multicore BPF method, incorporating custom layers that align with the particle filter approach.
    """
    model = Sequential()

    # Input layer with variable-length sequences
    model.add(tf.keras.Input(shape=input_shape))

    # Masking layer to handle padding
    model.add(tf.keras.layers.Masking(mask_value=0.0))

    # Convolutional layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.5))

    # LSTM layers for capturing temporal dynamics
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.5))

    # Custom layer for estimating source locations and waveforms
    model.add(MulticoreBPFLayer(num_particles=100, transition_matrix=transition_matrix, process_noise_cov=process_noise_cov, forward_matrix=forward_matrix))

    # Output layer for source locations (3D coordinates) and waveforms
    model.add(Dense(3, activation='linear'))  # Change output layer to produce 3 values per sample with linear activation

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[])

    return model

def load_preprocessed_data(data, labels, forward_matrices, state_transition_model, transition_matrix, process_noise_cov, n_components=4):
    """
    Load preprocessed EEG data, apply time-dependent spatial filtering, normalize it, and apply CSP and LDA.

    Args:
    data (np.ndarray): Pre-loaded EEG data.
    labels (np.ndarray): Array of labels corresponding to the data.
    forward_matrices (np.ndarray): Precomputed forward matrices for the head model.
    state_transition_model (function): State transition model function.
    transition_matrix (np.ndarray): State transition matrix.
    process_noise_cov (np.ndarray): Process noise covariance matrix.
    n_components (int): Number of CSP components to keep.

    Returns:
    np.ndarray: Loaded, normalized, and processed data as a 2D NumPy array (timesteps, features).

    Note:
    This function integrates the beamforming algorithm for generating time-dependent spatial filter matrices.
    The subsequent processing steps (normalization, CSP, and LDA) operate on the spatially filtered data produced
    by the time-dependent beamforming approach.
    """
    # Validate labels
    if len(labels) != data.shape[0]:
        raise ValueError("The number of labels must match the number of samples in the data.")

    # Generate beamforming matrices
    filter_matrices = generate_beamforming_matrix(data, forward_matrices, state_transition_model, transition_matrix, process_noise_cov)

    # Apply spatial filter
    filtered_data = apply_spatial_filter(data, filter_matrices)

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(filtered_data)

    # Apply CSP
    csp = CSP(n_components=n_components)
    csp_features = csp.fit_transform(normalized_data, labels)

    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    lda_features = lda.fit_transform(csp_features, labels)

    return lda_features

def train_model(model, data, labels, epochs=10, batch_size=32, validation_split=0.2):
    """
    Train the deep learning model.

    Args:
    model (tf.keras.Model): Compiled deep learning model.
    data (np.ndarray): Training data (timesteps, features).
    labels (np.ndarray): Training labels.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    validation_split (float): Fraction of the training data to be used as validation data.

    Returns:
    tf.keras.callbacks.History: Training history.
    """
    if validation_split > 0.0:
        history = model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    else:
        history = model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    return history

def predict(model, data):
    """
    Make predictions using the trained model and post-process the output for hologram generation.

    Args:
    model (tf.keras.Model): Trained deep learning model.
    data (np.ndarray): Data to make predictions on (timesteps, features).

    Returns:
    np.ndarray: Model predictions for hologram parameters.

    Note:
    This function will be updated to handle outputs from an enhanced model that includes
    source localization estimates, aligning with the advanced deep learning model requirements
    for the deep-brain robotics project.
    """
    predictions = model.predict(data)

    # Post-process the predictions to ensure they are suitable for hologram generation
    predictions = np.clip(predictions, 0, 1)

    return predictions

if __name__ == "__main__":
    # Load the generated sample data
    sample_data_file = 'sample_data.csv'
    sample_data_df = pd.read_csv(sample_data_file)
    data = sample_data_df.values

    # Load the processed labels
    labels_file = 'processed_labels.csv'
    labels_df = pd.read_csv(labels_file)
    labels = labels_df['Label'].values

    # Example forward matrices (replace with actual forward matrices)
    forward_matrices = np.random.rand(10, 64, 64)  # Example shape (num_matrices, channels, channels)

    # Initialize the transition matrix and process noise covariance matrix
    transition_matrix = np.eye(3)  # Example transition matrix (identity matrix)
    process_noise_cov = np.eye(3) * 0.1  # Example process noise covariance matrix

    # Example forward matrix (replace with actual forward matrix)
    forward_matrix = np.random.rand(100, 3)  # Example shape (num_particles, 3)

    # Reshape data to (samples, timesteps, features)
    data = data.reshape((data.shape[0], data.shape[1], 1))  # Assuming 1 feature per timestep for now
    input_shape = (data.shape[1], data.shape[2])  # Ensure input shape is 3D: (timesteps, features)
    model = create_deep_learning_model(input_shape, transition_matrix, process_noise_cov, forward_matrix)
    model.summary()

    # Train the model
    history = train_model(model, data, labels)

    # Make predictions
    predictions = predict(model, data)
    print("Predictions:", predictions)
