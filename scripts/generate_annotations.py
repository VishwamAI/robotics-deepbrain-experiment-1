import mne
import pandas as pd
import h5py
import os

def generate_annotations(eeg_file, output_dir):
    """
    Generate annotation files from EEG data.

    Args:
    eeg_file (str): Path to the input EEG data file.
    output_dir (str): Directory to save the generated annotation files.

    Returns:
    None
    """
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the EEG data from MATLAB file using h5py
        with h5py.File(eeg_file, 'r') as f:
            if 'EEGMMIDB' not in f.keys() or 'Signal' not in f['EEGMMIDB'].keys():
                raise KeyError("Expected keys 'EEGMMIDB' or 'Signal' not found in the HDF5 file.")

            signal_group = f['EEGMMIDB']['Signal']
            for subject in signal_group.keys():
                ref = signal_group[subject][0, 0]
                raw_data = f[ref][:]  # Adjust this key based on the actual structure of the MATLAB file

                # Extract sampling frequency from metadata if available
                sfreq = 256  # Default value
                if 'Frequency' in f['EEGMMIDB'].keys():
                    sfreq = f['EEGMMIDB']['Frequency'][0, 0]

                # Create an MNE Raw object from the loaded data
                ch_names = ['EEG' + str(i) for i in range(raw_data.shape[0])]
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq)  # Adjust channel names and sampling frequency as needed
                raw = mne.io.RawArray(raw_data, info)

                # Extract event markers
                events = mne.find_events(raw)

                # Create a DataFrame with the events
                event_df = pd.DataFrame(events, columns=['timestamp', 'event_marker', 'label'])

                # Save to a CSV file
                output_file = f"{output_dir}/{subject}_annotations.csv"
                event_df.to_csv(output_file, index=False)
                print(f"Annotation file saved: {output_file}")

    except FileNotFoundError:
        print(f"File not found: {eeg_file}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    eeg_file = '/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/MATLAB structure/EEGMMIDB_Curated.mat'
    output_dir = '/home/ubuntu/robotics-deepbrain-experiment-1/annotations'
    generate_annotations(eeg_file, output_dir)
