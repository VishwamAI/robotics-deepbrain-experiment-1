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

        # Check for NaNs or infinite values in the initial sample data
        if sample_df.isnull().values.any() or np.isinf(sample_df.values).any():
            print("Initial sample data contains NaNs or infinite values. Handling them...")
            sample_df = sample_df.apply(lambda x: x.fillna(x.mean()))  # Replace NaNs with the mean of the column
            sample_df = sample_df.apply(lambda x: x.replace([np.inf, -np.inf], x.mean()))  # Replace infinite values with the mean of the column

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

        # Check for NaNs or infinite values in the normalized data
        if normalized_df.isnull().values.any() or np.isinf(normalized_df.values).any():
            print("Normalized data contains NaNs or infinite values. Handling them...")
            normalized_df = normalized_df.apply(lambda x: x.fillna(x.mean()))  # Replace NaNs with the mean of the column
            normalized_df = normalized_df.apply(lambda x: x.replace([np.inf, -np.inf], x.mean()))  # Replace infinite values with the mean of the column

        # Extract band power features
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]  # Example frequency bands
        band_power_features = []
        for band in bands:
            low, high = band
            band_power = normalized_df.apply(lambda x: np.log(np.var(x[(x >= low) & (x <= high)]) + 1e-6) if np.var(x[(x >= low) & (x <= high)]) != 0 else 0, axis=1)
            band_power_features.append(band_power)
        band_power_df = pd.concat(band_power_features, axis=1)

        # Log the min, max, and mean values of the band power features
        print(f"Band power features - min: {band_power_df.min().min()}, max: {band_power_df.max().max()}, mean: {band_power_df.mean().mean()}")

        # Check for NaNs or infinite values in the band power features
        if band_power_df.isnull().values.any() or np.isinf(band_power_df.values).any():
            print("Band power features contain NaNs or infinite values. Handling them...")
            band_power_df = band_power_df.fillna(band_power_df.mean())  # Replace NaNs with the mean of the column
            band_power_df = band_power_df.replace([np.inf, -np.inf], band_power_df.mean())  # Replace infinite values with the mean of the column

        # Ensure there are multiple unique labels
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")  # Log the unique classes present in the labels array
        if len(unique_labels) < 2:
            raise ValueError("The labels file must contain at least two unique classes for CSP.")

        # Log the original shapes of the data and labels
        print(f"Original sample data shape: {sample_df.shape}")
        print(f"Original labels shape: {labels.shape}")
        print(f"Original unique labels: {np.unique(labels)}")

        n_samples = band_power_df.shape[0]
        n_channels = band_power_df.shape[1]
        remainder = n_samples % n_channels
        if remainder != 0:
            band_power_df = band_power_df.iloc[:-remainder]

        # Ensure labels match the number of trials
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        balanced_labels = []
        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        for i in range(min_count):
            for label in unique_labels:
                balanced_labels.append(labels[label_indices[label][i]])
        labels = np.array(balanced_labels)

        # Recalculate n_trials based on the length of the balanced labels array
        n_trials = len(labels) // n_channels
        if n_trials <= 0:
            raise ValueError("The number of trials must be greater than zero after trimming.")

        # Trim band_power_df to match the new number of trials
        band_power_df = band_power_df.iloc[:n_trials * n_channels]

        # Ensure the length of the balanced labels array matches the number of samples in the trimmed band_power_df
        if len(labels) != len(band_power_df):
            raise ValueError("The length of the balanced labels array does not match the number of samples in the trimmed band_power_df.")

        # Log the labels array after trimming
        print(f"Labels after trimming: {labels}")

        # Log the shapes and unique labels after trimming
        print(f"Labels shape after trimming: {labels.shape}")
        print(f"Unique labels after trimming: {np.unique(labels)}")

        unique_labels_after_trim = np.unique(labels)
        if len(unique_labels_after_trim) < 2:
            raise ValueError("The trimmed labels must contain at least two unique classes for CSP.")
        if len(unique_labels_after_trim) != len(unique_labels):
            raise ValueError(f"The number of unique labels after trimming ({len(unique_labels_after_trim)}) does not match the expected count ({len(unique_labels)}).")

        band_power_3d = band_power_df.values.reshape((n_trials, n_channels, -1))  # Reshape to (trials, channels, time)

        # Trim the labels array to match the new number of trials
        labels = labels[:n_trials]

        # Log the shapes of the reshaped data and labels
        print(f"Band power 3D shape: {band_power_3d.shape}")
        print(f"Labels shape after trimming: {labels.shape}")
        print(f"Number of trials: {n_trials}")
        print(f"Unique labels after trimming: {unique_labels_after_trim}")

        # Check for NaNs or infinite values in the data
        if np.any(np.isnan(band_power_3d)) or np.any(np.isinf(band_power_3d)):
            raise ValueError("band_power_3d contains NaNs or infinite values.")

        # Apply CSP
        csp = CSP(n_components=4)
        csp_features = csp.fit_transform(band_power_3d, labels)

        # Reduce dimensionality
        pca = PCA(n_components=10)
        reduced_data = pca.fit_transform(csp_features)
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
