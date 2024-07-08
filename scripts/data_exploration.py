import pandas as pd
import os

def load_csv_data(file_path, chunk_size=10000):
    """
    Load CSV data from the specified file path into a DataFrame.

    Args:
    file_path (str): Path to the CSV file.
    chunk_size (int): Number of rows per chunk to read at a time.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    try:
        # Read the CSV file in chunks to handle large files
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def explore_data(df):
    """
    Perform basic exploratory data analysis on the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    None
    """
    if df is not None:
        print("DataFrame Shape:", df.shape)
        print("First 5 Rows:\n", df.head())
        print("DataFrame Info:\n", df.info())
        print("DataFrame Description:\n", df.describe())
    else:
        print("DataFrame is None. Skipping exploration.")

if __name__ == "__main__":
    # Define the directory containing the CSV files
    csv_dir = 'Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files/'

    # List of sample CSV files to load and explore
    sample_files = [
        'SUB_001_SIG_01.csv',
        'SUB_001_SIG_02.csv',
        'SUB_001_SIG_03.csv'
    ]

    # Load and explore each sample CSV file
    for file_name in sample_files:
        file_path = os.path.join(csv_dir, file_name)
        print(f"Exploring {file_name}...")
        df = load_csv_data(file_path)
        explore_data(df)
        print("\n" + "="*50 + "\n")
