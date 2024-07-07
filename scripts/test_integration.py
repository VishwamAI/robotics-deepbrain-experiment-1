import os
import numpy as np
import cv2
from deep_learning_model import create_deep_learning_model, load_preprocessed_data, predict, train_model
from image_processing import load_image, preprocess_image, generate_hologram, display_image

# Define the input shape for the model
input_shape = (1, 19679, 64)

# Create the deep learning model
model = create_deep_learning_model(input_shape)
model.summary()

# Load preprocessed EEG data
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
history = train_model(model, data, labels, validation_split=0.0)  # Set validation_split to 0.0 to avoid splitting

# Make predictions using the trained model
predictions = predict(model, data)

# Load and preprocess a sample image
image_path = "/home/ubuntu/browser_downloads/codioful-formerly-gradienta-7E5kq_sW0Ew-unsplash.jpg"
try:
    image = load_image(image_path)
    preprocessed_image = preprocess_image(image)
except FileNotFoundError as e:
    print(e)
    exit(1)

# Generate a hologram using the model's predictions
hologram = generate_hologram(preprocessed_image, predictions)

# Display the generated hologram
display_image(hologram, window_name="Hologram")
