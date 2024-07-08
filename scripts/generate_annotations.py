import mne
import pandas as pd
import h5py
import os
import numpy as np

def process_event_data(event_data, index, f, strategy='mean'):
    """
    Process event data element by dereferencing and handling multi-element arrays.

    Args:
    event_data (list): List of event data elements.
    index (int): Index of the element to process.
    f (h5py.File): HDF5 file object.
    strategy (str): Strategy to handle multi-element arrays ('mean' or 'first').

    Returns:
    Processed event data element.
    """
    if isinstance(event_data[index], h5py.Reference):
        print(f"Dereferencing event_data[{index}]: {event_data[index]}")
        event_data[index] = f[event_data[index]][()]
        print(f"Dereferenced event_data[{index}]: {event_data[index]}")
        if isinstance(event_data[index], np.ndarray):
            print(f"Size of event_data[{index}] before conversion: {event_data[index].size}")
            if event_data[index].size == 1:
                event_data[index] = event_data[index].item()  # Convert single-element array to scalar
            else:
                print(f"event_data[{index}] is not a single-element array: {event_data[index]}")
                # Handle the case where event_data[index] is not a single-element array
                # For now, log the size and content of the array
                print(f"event_data[{index}] size: {event_data[index].size}, content: {event_data[index]}")
                # Decide on a strategy to handle multi-element arrays
                if strategy == 'mean':
                    try:
                        event_data[index] = np.mean(event_data[index])
                        print(f"Applied 'mean' strategy to event_data[{index}]: {event_data[index]}")
                    except Exception as e:
                        print(f"Error calculating mean of event_data[{index}]: {e}")
                        # Fallback strategy: take the first element
                        event_data[index] = event_data[index][0]
                        print(f"Fallback to first element for event_data[{index}]: {event_data[index]}")
                elif strategy == 'first':
                    event_data[index] = event_data[index][0]
                    print(f"Applied 'first' strategy to event_data[{index}]: {event_data[index]}")
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
        print(f"Final event_data[{index}]: {event_data[index]}")
    else:
        print(f"event_data[{index}] is not a reference: {event_data[index]}")
    return event_data[index]

def generate_annotations(eeg_file, output_dir, strategy='mean'):
    """
    Generate annotation files from EEG data.

    Args:
    eeg_file (str): Path to the input EEG data file.
    output_dir (str): Directory to save the generated annotation files.
    strategy (str): Strategy to handle multi-element arrays ('mean' or 'first').

    Returns:
    None
    """
    version = "1.0.1"
    print(f"Starting generate_annotations (version {version}) with eeg_file: {eeg_file}, output_dir: {output_dir}, and strategy: {strategy}")
    print("Script execution started.")

    # Validate strategy parameter
    if strategy not in ['mean', 'first']:
        raise ValueError(f"Invalid strategy: {strategy}. Valid options are 'mean' or 'first'.")

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
                raw_data = signal_group[subject][:]
                print(f"Raw data shape for subject {subject}: {raw_data.shape}")
                print(f"Raw data content for subject {subject}: {raw_data}")

                # Ensure raw_data is fully dereferenced
                for i in range(raw_data.shape[0]):
                    for j in range(raw_data.shape[1]):
                        if isinstance(raw_data[i, j], h5py.Reference):
                            raw_data[i, j] = f[raw_data[i, j]][()]
                            if isinstance(raw_data[i, j], np.ndarray):
                                raw_data[i, j] = raw_data[i, j].item()  # Convert single-element array to scalar

                print(f"Dereferenced raw data content for subject {subject}: {raw_data}")
                print(f"Type of raw_data: {type(raw_data)}, shape: {raw_data.shape}")

                # Extract sampling frequency from metadata if available
                sfreq = 256  # Default value
                if 'Frequency' in f['EEGMMIDB'].keys():
                    sfreq = f['EEGMMIDB']['Frequency'][0, 0]
                print(f"Sampling frequency: {sfreq}")

                # Create an MNE Raw object from the loaded data
                ch_names = ['EEG' + str(i) for i in range(raw_data.shape[0])]
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq)  # Adjust channel names and sampling frequency as needed
                print(f"Channel names: {ch_names}")
                print(f"Info object: {info}")
                raw = mne.io.RawArray(raw_data, info)
                print(f"Created MNE Raw object for subject {subject}")

                # Check if the stim channel exists
                if 'STI 014' in raw.ch_names:
                    stim_channel = 'STI 014'
                else:
                    # Create a synthetic stim channel based on annotations
                    annotations_key = 'Annotations'
                    if annotations_key not in f['EEGMMIDB'].keys():
                        print(f"Annotations key not found for subject: {subject}")
                        continue

                    print(f"Annotations key found for subject: {subject}")
                    print(f"Attempting to access annotations for subject: {subject}")
                    try:
                        annotations_refs = f['EEGMMIDB']['Annotations'][subject][0]
                        print(f"Annotations references for subject {subject}: {annotations_refs}")
                        try:
                            stim_data = np.zeros(raw_data.shape[1])
                            print(f"Initialized stim_data array with shape: {stim_data.shape}, type: {type(stim_data)}")
                        except Exception as e:
                            print(f"Error creating stim_data array: {e}")
                            continue
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
                                                            if len(event_data) < 3:
                                                                print(f"Unexpected length of event_data: {len(event_data)}, value: {event_data}")
                                                                raise ValueError(f"Unexpected length of event_data: {len(event_data)}")

                                                            print(f"Type of event_data[2] before dereferencing: {type(event_data[2])}, value: {event_data[2]}")
                                                            print(f"Type of event_data[0] before dereferencing: {type(event_data[0])}, value: {event_data[0]}")
                                                            print(f"Type of event_data[1] before dereferencing: {type(event_data[1])}, value: {event_data[1]}")
                                                            try:
                                                                # Process event_data elements
                                                                try:
                                                                    event_data[2] = process_event_data(event_data, 2, f, strategy)
                                                                    event_data[0] = process_event_data(event_data, 0, f, strategy)
                                                                    event_data[1] = process_event_data(event_data, 1, f, strategy)
                                                                except ValueError as ve:
                                                                    print(f"ValueError encountered while processing event_data: {ve}")
                                                                    continue  # Skip to the next annotation if an invalid strategy is encountered
                                                                except Exception as e:
                                                                    print(f"Error processing event_data: {e}")
                                                                    raise
                                                            except Exception as e:
                                                                print(f"Error dereferencing event_data: {e}")
                                                                print(f"Type of event_data[0] after error: {type(event_data[0])}, value: {event_data[0]}")
                                                                print(f"Type of event_data[1] after error: {type(event_data[1])}, value: {event_data[1]}")
                                                                print(f"Type of event_data[2] after error: {type(event_data[2])}, value: {event_data[2]}")

                                                            print(f"Event data[2] after dereferencing: {event_data[2]}")
                                                            print(f"Type of event_data[0]: {type(event_data[0])}, value: {event_data[0]}")
                                                            print(f"Type of event_data[1]: {type(event_data[1])}, value: {event_data[1]}")
                                                            if not isinstance(event_data[2], (int, float)):
                                                                print(f"Unexpected data type for event_data[2]: {type(event_data[2])}, value: {event_data[2]}")
                                                                raise ValueError(f"Unexpected data type for event_data[2]: {type(event_data[2])}")
                                                            if not isinstance(event_data[0], (int, float)) or not isinstance(event_data[1], (int, float)):
                                                                raise ValueError(f"Unexpected data type for event_data[0] or event_data[1]: {type(event_data[0])}, {type(event_data[1])}")
                                                            print(f"Assigning event_data[2] to stim_data[{int(event_data[0])}:{int(event_data[1])}]")
                                                            try:
                                                                start_idx = int(event_data[0])
                                                                end_idx = int(event_data[1])
                                                                if start_idx < 0 or end_idx > len(stim_data):
                                                                    raise IndexError(f"Index out of bounds: start_idx={start_idx}, end_idx={end_idx}, stim_data length={len(stim_data)}")
                                                                print(f"stim_data before assignment: {stim_data[start_idx:end_idx]}")
                                                                stim_data[start_idx:end_idx] = event_data[2]
                                                                print(f"stim_data after assignment: {stim_data[start_idx:end_idx]}, assigned value: {event_data[2]}")
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
    strategy = 'mean'  # Default strategy
    try:
        generate_annotations(eeg_file, output_dir, strategy)
    except ValueError as ve:
        print(f"ValueError encountered: {ve}")
