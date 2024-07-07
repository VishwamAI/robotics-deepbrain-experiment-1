import os
import numpy as np
import cv2
from deep_learning_model import create_deep_learning_model, load_preprocessed_data, predict
from image_processing import load_image, preprocess_image, generate_hologram, display_image

# Define the input shape for the model
input_shape = (256, 64)

# Create the deep learning model
model = create_deep_learning_model(input_shape)
model.summary()

# Load preprocessed EEG data
csv_dir = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/'
sample_file = 'SUB_001_SIG_01.csv'
file_path = os.path.join(csv_dir, sample_file)
data = load_preprocessed_data(file_path)

# Make predictions using the trained model
predictions = predict(model, data)

# Load and preprocess a sample image
image_path = "/home/ubuntu/browser_downloads/codioful-formerly-gradienta-7E5kq_sW0Ew-unsplash.jpg"
image = load_image(image_path)
preprocessed_image = preprocess_image(image)

# Generate a hologram using the model's predictions
hologram = generate_hologram(preprocessed_image, predictions)

# Display the generated hologram
display_image(hologram, window_name="Hologram")
