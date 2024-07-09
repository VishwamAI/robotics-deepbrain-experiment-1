import os
import numpy as np
import cv2
from deep_learning_model import create_deep_learning_model, load_preprocessed_data, predict, train_model
from image_processing import load_image, preprocess_image, generate_hologram
from data_preprocessing import load_csv_data_in_chunks, normalize_data, segment_data

# Initialize the transition matrix and process noise covariance matrix
transition_matrix = np.eye(3)  # Example transition matrix (identity matrix)
process_noise_cov = np.eye(3) * 0.1  # Example process noise covariance matrix

# Example forward matrix (replace with actual forward matrix)
forward_matrix = np.random.rand(100, 3).astype(np.float64)  # Example shape (num_particles, 3) and cast to float64

# Create the deep learning model
model = create_deep_learning_model((128, 64), transition_matrix, process_noise_cov, forward_matrix)
model.summary()

# Load preprocessed EEG data
csv_dir = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/'
sample_file = 'SUB_001_SIG_01.csv'
file_path = os.path.join(csv_dir, sample_file)
data = load_csv_data_in_chunks(file_path)

if data is not None:
    # Normalize the data
    normalized_data = normalize_data(data)

    # Segment the data
    segments = segment_data(normalized_data)

    # Reshape data to match the model's expected input shape
    data = np.array([segment.values for segment in segments])
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2]))

    # Placeholder function to generate labels based on EEG data
    def generate_labels(data):
        num_samples = data.shape[0]
        labels = {
            "position": np.random.rand(num_samples, 2),  # Generate random positions
            "intensity": np.random.rand(num_samples)    # Generate random intensities
        }
        return labels

    # Generate labels for the data
    labels = generate_labels(data)

    # Train the model
    history = train_model(model, data, labels, validation_split=0.0)  # Set validation_split to 0.0 to avoid splitting

    # Make predictions using the trained model
    predictions = predict(model, data)

    # Print debugging information
    print("Predictions shape:", predictions.shape)
    print("Predictions type:", type(predictions))
    print("Predictions:", predictions)

    # Add assertions to validate outcomes
    assert predictions.shape[0] == data.shape[0], "Mismatch in number of predictions and input data samples"
    assert predictions.shape[1] == 3, "Predictions should have 3 columns for position and intensity"

    # Load and preprocess a sample image
    image_path = "/home/ubuntu/browser_downloads/codioful-formerly-gradienta-7E5kq_sW0Ew-unsplash.jpg"
    try:
        image = load_image(image_path)
        preprocessed_image = preprocess_image(image)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Generate a hologram using the model's predictions
    position = predictions[:, :2]
    intensity = predictions[:, 2]
    hologram = generate_hologram(preprocessed_image, position, intensity)

    # Save the generated hologram to a file
    hologram_output_path = "/home/ubuntu/robotics-deepbrain-experiment-1/output/hologram.png"
    cv2.imwrite(hologram_output_path, hologram)
    print(f"Hologram saved to {hologram_output_path}")

    # Remove the call to display the generated hologram
    # display_image(hologram, window_name="Hologram")
else:
    print(f"File not found: {file_path}")
