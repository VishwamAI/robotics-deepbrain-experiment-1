import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.decoding import CSP
import os

def generate_sample_data(input_dir, output_file, labels_file, sample_size=1000, chunk_size=10000):
    """
    Generate a sample dataset from the input EEG data files.

    Args:
    input_dir (str): Directory containing the input EEG data files.
    output_file (str): Path to save the generated sample data.
    labels_file (str): Path to the processed labels file.
    sample_size (int): Number of samples to include in the generated dataset.
    chunk_size (int): Number of rows to read at a time from the input files.

    Returns:
    None
    """
    try:
        # Initialize an empty DataFrame to store the sample data
        sample_df = pd.DataFrame()

        # Load processed labels
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['Label'].values

        # Iterate over signal files in the input directory
        for filename in os.listdir(input_dir):
            if "_sig" in filename.lower() and filename.lower().endswith(".csv"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")

                # Read the data in chunks
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    sample_df = pd.concat([sample_df, chunk])
                    if len(sample_df) >= sample_size:
                        break

                if len(sample_df) >= sample_size:
                    break

        # Trim the sample data to the desired sample size
        sample_df = sample_df.head(sample_size)

        # Ensure the labels match the sample data length
        if len(labels) < len(sample_df):
            labels = np.resize(labels, len(sample_df))
        elif len(labels) > len(sample_df):
            labels = labels[:len(sample_df)]

        # Log the shapes of the data and labels
        print(f"Sample data shape: {sample_df.shape}")
        print(f"Labels shape: {labels.shape}")

        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(sample_df)
        normalized_df = pd.DataFrame(normalized_data, columns=sample_df.columns)

        # Extract band power features
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]  # Example frequency bands
        band_power_features = []
        for band in bands:
            low, high = band
            band_power = normalized_df.apply(lambda x: np.log(np.var(x[(x >= low) & (x <= high)]) + 1e-10), axis=1)
            band_power_features.append(band_power)
        band_power_df = pd.concat(band_power_features, axis=1)

        # Ensure there are multiple unique labels
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")  # Log the unique classes present in the labels array
        if len(unique_labels) < 2:
            raise ValueError("The labels file must contain at least two unique classes for CSP.")

        # Reshape data for CSP
        n_samples = band_power_df.shape[0]
        n_channels = band_power_df.shape[1]
        n_trials = n_samples // len(unique_labels)  # Adjust for multiple classes
        remainder = n_samples % len(unique_labels)
        if remainder != 0:
            band_power_df = band_power_df.iloc[:-remainder]
            labels = labels[:n_trials * len(unique_labels)]
        labels = labels[:n_trials * len(unique_labels)]  # Ensure labels match the number of trials
        unique_labels_after_trim = np.unique(labels)
        if len(unique_labels_after_trim) < 2:
            raise ValueError("The trimmed labels must contain at least two unique classes for CSP.")
        band_power_3d = band_power_df.values.reshape((n_trials, len(unique_labels), n_channels))

        # Log the shapes of the reshaped data and labels
        print(f"Band power 3D shape: {band_power_3d.shape}")
        print(f"Labels shape after trimming: {labels.shape}")

        # Apply CSP
        csp = CSP(n_components=4)
        csp_features = csp.fit_transform(band_power_3d, labels)
        csp_df = pd.DataFrame(csp_features)

        # Reduce dimensionality
        pca = PCA(n_components=10)
        reduced_data = pca.fit_transform(csp_df)
        reduced_df = pd.DataFrame(reduced_data)

        # Save the generated sample data
        reduced_df.to_csv(output_file, index=False)
        print(f"Sample data generated and saved to {output_file}")

    except FileNotFoundError:
        print(f"File not found: {input_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = '/home/ubuntu/robotics-deepbrain-experiment-1/datasets'
    output_file = 'sample_data.csv'
    labels_file = 'processed_labels.csv'
    generate_sample_data(input_dir, output_file, labels_file)
