# Project Documentation: Deepbrain-Experiment-1

## Project Overview
The Deepbrain-Experiment-1 project aims to develop an advanced deep learning model for a deep-brain robotics project. The project includes the creation of a 2D imagination hologram experience that interacts with the brain, leveraging EEG data for brain signal interpretation.

## Development Environment
The development environment was set up using a Python virtual environment. Essential packages were installed, including TensorFlow, OpenCV, MNE, and other necessary libraries.

## Folder Structure
The project is organized into the following directories:
- `datasets/`: Contains the EEG dataset used for the project.
- `scripts/`: Contains Python scripts for data preprocessing, image processing, neural network model, and integration testing.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `models/`: Saved models and checkpoints.
- `documentation/`: Project documentation and references.

## Data
The EEG dataset `dpmtgrn8d8-4.zip` was downloaded from Mendeley Data. It includes a MATLAB structure file (`EEGMMIDB_Curated.mat`) and a RAR archive containing CSV files (`eegmmidb.rar`). The data was preprocessed to extract relevant features for model training.

## Scripts
### `data_preprocessing.py`
This script handles the preprocessing of EEG data, including band power feature extraction and the application of Common Spatial Patterns (CSP).

### `image_processing.py`
This script contains the `generate_hologram` function, which generates a 2D hologram based on the model's output.

### `deep_learning_model.py`
This script defines the deep learning model architecture, including convolutional layers for feature extraction, LSTM layers for capturing temporal dynamics, and fully connected layers for final prediction. It also integrates CSP and LDA for enhanced EEG data preprocessing.

### `test_integration.py`
This script performs integration testing, ensuring that the model and hologram generation process work as intended. It includes steps to save the generated hologram image to a file.

## Neural Network Model
The deep learning model is designed to interpret EEG data and generate a 2D hologram. The model architecture includes:
- Convolutional layers for feature extraction.
- LSTM layers for capturing temporal dynamics.
- Fully connected layers for final prediction.
- Integration of CSP and LDA for enhanced EEG data preprocessing.

## Hologram Experience
The hologram experience is generated using the `generate_hologram` function in the `image_processing.py` script. The function takes an input image and a model output dictionary containing 'position' and 'intensity' keys, and generates a 2D hologram by drawing a circle on a blank canvas, with the position determined by the model output and scaled to the image dimensions.

## Challenges and Solutions
### OpenCV Circle Function Error
An error with the `cv2.circle` function was resolved by ensuring the `intensity` value is a scalar.

### Qt Platform Plugin "xcb" Loading Error
This error was resolved by modifying the `test_integration.py` script to remove the call to `display_image`, which was causing the error in the headless environment.

### 'mne' Module Not Found Error
The 'mne' module not found error was resolved by installing the missing package in the virtual environment.

## References
- EEG dataset: [Mendeley Data](https://data.mendeley.com/)
- Research papers and articles on EEG signal processing and brain-computer interfaces.
- TensorFlow, OpenCV, MNE, and other libraries used in the project.

This documentation provides a comprehensive overview of the development process, including the setup, scripts, model architecture, and challenges encountered. It serves as a guide for understanding the project's structure and functionality.
