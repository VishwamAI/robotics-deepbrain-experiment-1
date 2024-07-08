import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.decoding import CSP
import os

def generate_sample_data(input_file, output_file, labels_file, sample_size=1000, chunk_size=10000):
    """
    Generate a sample dataset from the input EEG data file.

    Args:
    input_file (str): Path to the input EEG data file.
    output_file (str): Path to save the generated sample data.
    labels_file (str): Path to the processed labels file.
    sample_size (int): Number of samples to include in the generated dataset.
    chunk_size (int): Number of rows to read at a time from the input file.

    Returns:
    None
    """
    try:
        # Initialize an empty DataFrame to store the sample data
        sample_df = pd.DataFrame()

        # Read the data in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            sample_df = pd.concat([sample_df, chunk])
            if len(sample_df) >= sample_size:
                break

        # Trim the sample data to the desired sample size
        sample_df = sample_df.head(sample_size)

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

        # Load processed labels
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['Label'].values[:band_power_df.shape[0]]

        # Ensure there are multiple unique labels
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            raise ValueError("The labels file must contain at least two unique classes for CSP.")

        # Reshape data for CSP
        n_samples = band_power_df.shape[0]
        n_channels = band_power_df.shape[1]
        n_trials = n_samples // len(unique_labels)  # Adjust for multiple classes
        band_power_3d = band_power_df.values.reshape((n_trials, len(unique_labels), n_channels))

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
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/SUB_001_SIG_01.csv'
    output_file = 'sample_data.csv'
    labels_file = 'processed_labels.csv'
    generate_sample_data(input_file, output_file, labels_file)
