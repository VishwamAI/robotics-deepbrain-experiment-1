import mne
import pandas as pd
from scipy.io import loadmat

def generate_annotations(eeg_file, output_file):
    """
    Generate annotation files from EEG data.

    Args:
    eeg_file (str): Path to the input EEG data file.
    output_file (str): Path to save the generated annotation file.

    Returns:
    None
    """
    try:
        # Load the EEG data from MATLAB file
        mat = loadmat(eeg_file)
        raw_data = mat['EEG']  # Adjust this key based on the actual structure of the MATLAB file

        # Create an MNE Raw object from the loaded data
        info = mne.create_info(ch_names=['EEG'], sfreq=256)  # Adjust channel names and sampling frequency as needed
        raw = mne.io.RawArray(raw_data, info)

        # Extract event markers
        events = mne.find_events(raw)

        # Create a DataFrame with the events
        event_df = pd.DataFrame(events, columns=['timestamp', 'event_marker', 'label'])

        # Save to a CSV file
        event_df.to_csv(output_file, index=False)
    except FileNotFoundError:
        print(f"File not found: {eeg_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    eeg_file = '/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/MATLAB structure/EEGMMIDB_Curated.mat'
    output_file = '/home/ubuntu/robotics-deepbrain-experiment-1/annotations.csv'
    generate_annotations(eeg_file, output_file)
