import h5py
import pandas as pd
import numpy as np
import os
import logging

def generate_eeg_annotations(eeg_file_path, output_csv_path):
    """
    Generates annotation files for the EEG dataset.

    Args:
    eeg_file_path (str): Path to the EEG data file.
    output_csv_path (str): Path to the output CSV file for annotations.

    Returns:
    None
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path))

        with h5py.File(eeg_file_path, 'r') as f:
            annotations_group = f['EEGMMIDB']['Annotations']
            events = []
            for subject in annotations_group.keys():
                logging.info(f"Processing annotations for {subject}")
                annotations_refs = annotations_group[subject][0]
                for ref in annotations_refs:
                    if isinstance(ref, h5py.Reference):
                        dereferenced_data = f[ref]
                        for sub_ref in dereferenced_data:
                            if isinstance(sub_ref, np.ndarray):
                                for actual_ref in sub_ref:
                                    if isinstance(actual_ref, h5py.Reference):
                                        sub_dereferenced_data = f[actual_ref]
                                        if isinstance(sub_dereferenced_data, h5py.Dataset):
                                            event_data = np.array(sub_dereferenced_data[:])
                                            logging.info(f"Event data shape for {subject}: {event_data.shape}")
                                            for event in event_data:
                                                if len(event) == 1:
                                                    # Assuming single-element events represent timestamps
                                                    events.append([subject, event[0], None, None])
                                                elif len(event) == 2:
                                                    # Assuming two-element events represent timestamp and event_marker
                                                    events.append([subject, event[0], event[1], None])
                                                elif len(event) == 3:
                                                    # Assuming three-element events represent timestamp, event_marker, and label
                                                    events.append([subject] + list(event))
                                                else:
                                                    logging.warning(f"Unexpected event data length for {subject}: {event}")
                                        else:
                                            logging.warning(f"Unexpected data type: {type(sub_dereferenced_data)}")
                                    else:
                                        logging.warning(f"Unexpected reference type: {type(actual_ref)}")
                            else:
                                logging.warning(f"Unexpected reference type: {type(sub_ref)}")
                    else:
                        logging.warning(f"Unexpected reference type: {type(ref)}")

            # Create a DataFrame with the events
            event_df = pd.DataFrame(events, columns=['subject', 'timestamp', 'event_marker', 'label'])

            # Save to a CSV file
            event_df.to_csv(output_csv_path, index=False)
            logging.info(f"Annotations saved to {output_csv_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    eeg_file_path = '/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/MATLAB structure/EEGMMIDB_Curated.mat'
    output_csv_path = '/home/ubuntu/robotics-deepbrain-experiment-1/output/eeg_annotations.csv'
    generate_eeg_annotations(eeg_file_path, output_csv_path)
