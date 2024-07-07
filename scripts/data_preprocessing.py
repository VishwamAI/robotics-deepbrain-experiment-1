import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

            # Reduce dimensionality
            reduced_df = reduce_dimensionality(normalized_df)
            print("Reduced Dimensionality Data:\n", reduced_df.head())

            # Segment the data
            segments = segment_data(normalized_df, window_size=128, step_size=64)
            print("Number of Segments:", len(segments))
            print("First Segment:\n", segments[0])

            # Plot EEG signals
            plot_eeg_signals(normalized_df)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
