import mne
import pandas as pd
import h5py
import os
import numpy as np

def generate_annotations(eeg_file, output_dir):
    version = "1.0.1"
    print(f"Starting generate_annotations (version {version}) with eeg_file: {eeg_file} and output_dir: {output_dir}")
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
            print(f"Keys in HDF5 file: {list(f.keys())}")
            if 'EEGMMIDB' not in f.keys() or 'Signal' not in f['EEGMMIDB'].keys():
                raise KeyError("Expected keys 'EEGMMIDB' or 'Signal' not found in the HDF5 file.")

            print("Loaded HDF5 file successfully.")

            signal_group = f['EEGMMIDB']['Signal']
            for subject in signal_group.keys():
                print(f"Processing subject: {subject}")
                raw_data = signal_group[subject][:]  # Adjust this key based on the actual structure of the MATLAB file

                # Extract sampling frequency from metadata if available
                sfreq = 256  # Default value
                if 'Frequency' in f['EEGMMIDB'].keys():
                    sfreq = f['EEGMMIDB']['Frequency'][0, 0]
                print(f"Sampling frequency: {sfreq}")

                # Create an MNE Raw object from the loaded data
                ch_names = ['EEG' + str(i) for i in range(raw_data.shape[0])]
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq)  # Adjust channel names and sampling frequency as needed
                raw = mne.io.RawArray(raw_data, info)

                # Check if the stim channel exists
                if 'STI 014' in raw.ch_names:
                    stim_channel = 'STI 014'
                else:
                    # Create a synthetic stim channel based on annotations
                    annotations_key = 'Annotations'
                    if annotations_key not in f['EEGMMIDB'].keys():
                        print(f"Annotations key not found for subject: {subject}")
                        continue

                    try:
                        annotations_refs = f['EEGMMIDB']['Annotations'][subject][0]
                        print(f"Annotations references for subject {subject}: {annotations_refs}")
                        try:
                            stim_data = np.zeros(raw_data.shape[1])
                        except Exception as e:
                            print(f"Error creating stim_data array: {e}")
                        for ref in annotations_refs:
                            print(f"Type of ref: {type(ref)}, value: {ref}")
                            if isinstance(ref, h5py.Reference):
                                try:
                                    dereferenced_data = f[ref]
                                    print(f"Dereferenced data for ref: {ref}, type: {type(dereferenced_data)}, value: {dereferenced_data}")
                                    for sub_ref in dereferenced_data:
                                        print(f"Type of sub_ref: {type(sub_ref)}, value: {sub_ref}")
                                        if isinstance(sub_ref, np.ndarray):
                                            for actual_ref in sub_ref:
                                                print(f"Type of actual_ref: {type(actual_ref)}, value: {actual_ref}")
                                                if isinstance(actual_ref, h5py.Reference):
                                                    try:
                                                        sub_dereferenced_data = f[actual_ref]
                                                        print(f"Dereferenced data for actual_ref: {actual_ref}, type: {type(sub_dereferenced_data)}, value: {sub_dereferenced_data}")
                                                        if isinstance(sub_dereferenced_data, h5py.Dataset):
                                                            event_data = np.array(sub_dereferenced_data[:])
                                                            print(f"Event data: {event_data}")
                                                            print(f"Event data before assignment: {event_data}")
                                                            if isinstance(event_data[2], h5py.Reference):
                                                                try:
                                                                    print(f"Dereferencing event_data[2]: {event_data[2]}")
                                                                    event_data[2] = f[event_data[2]][()]
                                                                    print(f"Dereferenced event_data[2]: {event_data[2]}")
                                                                except Exception as e:
                                                                    print(f"Error dereferencing event_data[2]: {e}")
                                                            print(f"Event data[2] after dereferencing: {event_data[2]}")
                                                            print(f"Type of event_data[0]: {type(event_data[0])}, value: {event_data[0]}")
                                                            print(f"Type of event_data[1]: {type(event_data[1])}, value: {event_data[1]}")
                                                            print(f"Type of event_data[2]: {type(event_data[2])}, value: {event_data[2]}")
                                                            if not isinstance(event_data[2], (int, float)):
                                                                raise ValueError(f"Unexpected data type for event_data[2]: {type(event_data[2])}")
                                                            if not isinstance(event_data[0], (int, float)) or not isinstance(event_data[1], (int, float)):
                                                                raise ValueError(f"Unexpected data type for event_data[0] or event_data[1]: {type(event_data[0])}, {type(event_data[1])}")
                                                            print(f"Assigning event_data[2] to stim_data[{int(event_data[0])}:{int(event_data[1])}]")
                                                            try:
                                                                stim_data[int(event_data[0]):int(event_data[1])] = event_data[2]
                                                            except Exception as e:
                                                                print(f"Error assigning event_data[2] to stim_data: {e}")
                                                    except Exception as e:
                                                        print(f"Error dereferencing actual_ref: {e}")
                                except Exception as e:
                                    print(f"Error dereferencing ref: {e}")
                    except Exception as e:
                        print(f"Error processing annotations_refs: {e}")

                    raw.add_channels([mne.io.RawArray(stim_data[np.newaxis, :], mne.create_info(['STI 014'], sfreq))])
                    stim_channel = 'STI 014'

                # Extract event markers
                events = mne.find_events(raw, stim_channel=stim_channel)

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
