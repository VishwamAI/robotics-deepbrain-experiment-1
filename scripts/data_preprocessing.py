import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mne.decoding import CSP
import os
import matplotlib.pyplot as plt

def normalize_data(df):
    """
    Normalize the data using StandardScaler.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)

def reduce_dimensionality(df, n_components=10):
    """
    Reduce the dimensionality of the data using PCA.

    Args:
    df (pd.DataFrame): Input DataFrame.
    n_components (int): Number of principal components to keep.

    Returns:
    pd.DataFrame: DataFrame with reduced dimensionality.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(df)
    return pd.DataFrame(reduced_data)

def segment_data(df, window_size=128, step_size=64):
    """
    Segment the data into smaller time windows.

    Args:
    df (pd.DataFrame): Input DataFrame.
    window_size (int): Size of each window.
    step_size (int): Step size between windows.

    Returns:
    list: List of segmented DataFrames.
    """
    segments = []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        segment = df.iloc[start:end]
        segments.append(segment)
    return segments

def load_csv_data_in_chunks(file_path, chunk_size=10000):
    """
    Load CSV data from the specified file path in chunks.

    Args:
    file_path (str): Path to the CSV file.
    chunk_size (int): Number of rows per chunk to read at a time.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    try:
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def plot_eeg_signals(df, title="EEG Signals"):
    """
    Plot EEG signals from the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        plt.plot(df[column], label=column)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.show()

def extract_band_power_features(df, bands):
    """
    Extract band power features from EEG signals.

    Args:
    df (pd.DataFrame): Input DataFrame.
    bands (list of tuple): List of frequency bands as (low, high) tuples.

    Returns:
    pd.DataFrame: DataFrame with band power features.
    """
    band_power_features = []
    for band in bands:
        low, high = band
        band_power = df.apply(lambda x: np.log(np.var(x[(x >= low) & (x <= high)])), axis=1)
        band_power_features.append(band_power)
    return pd.concat(band_power_features, axis=1)

def apply_csp(df, labels, n_components=4):
    """
    Apply Common Spatial Patterns (CSP) to EEG signals.

    Args:
    df (pd.DataFrame): Input DataFrame.
    labels (np.ndarray): Array of labels corresponding to the data.
    n_components (int): Number of CSP components to keep.

    Returns:
    pd.DataFrame: DataFrame with CSP features.
    """
    csp = CSP(n_components=n_components)
    csp_features = csp.fit_transform(df.values, labels)
    return pd.DataFrame(csp_features)

if __name__ == "__main__":
    try:
        # Example usage
        csv_dir = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/'
        sample_file = 'SUB_001_SIG_01.csv'
        file_path = os.path.join(csv_dir, sample_file)

        # Load the data in chunks
        df = load_csv_data_in_chunks(file_path)

        if df is not None:
            # Normalize the data
            normalized_df = normalize_data(df)
            print("Normalized Data:\n", normalized_df.head())

            # Extract band power features
            bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]  # Example frequency bands
            band_power_df = extract_band_power_features(normalized_df, bands)
            print("Band Power Features:\n", band_power_df.head())

            # Apply CSP
            labels = np.random.randint(0, 2, size=(band_power_df.shape[0],))  # Example labels
            csp_df = apply_csp(band_power_df, labels)
            print("CSP Features:\n", csp_df.head())

            # Reduce dimensionality
            reduced_df = reduce_dimensionality(csp_df)
            print("Reduced Dimensionality Data:\n", reduced_df.head())

            # Segment the data
            segments = segment_data(reduced_df, window_size=128, step_size=64)
            print("Number of Segments:", len(segments))
            print("First Segment:\n", segments[0])

            # Plot EEG signals
            plot_eeg_signals(normalized_df)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
