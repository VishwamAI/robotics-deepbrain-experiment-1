import h5py
import numpy as np

def read_annotations(eeg_file):
    """
    Read and print annotation data from EEG file.

    Args:
    eeg_file (str): Path to the input EEG data file.

    Returns:
    None
    """
    with h5py.File(eeg_file, 'r') as f:
        annotations_refs = f['EEGMMIDB']['Annotations']['Subject_100_Annotations'][0]
        for ref in annotations_refs:
            if isinstance(ref, h5py.Reference):
                dereferenced_data = f[ref]
                for sub_ref in dereferenced_data:
                    if isinstance(sub_ref, np.ndarray):
                        for actual_ref in sub_ref:
                            if isinstance(actual_ref, h5py.Reference):
                                sub_dereferenced_data = f[actual_ref]
                                if isinstance(sub_dereferenced_data, h5py.Dataset):
                                    print(np.array(sub_dereferenced_data[:]))
                                else:
                                    print("Unexpected data type:", type(sub_dereferenced_data))
                            else:
                                print("Unexpected reference type:", type(actual_ref))
                    else:
                        print("Unexpected reference type:", type(sub_ref))
            else:
                print("Unexpected reference type:", type(ref))

if __name__ == "__main__":
    eeg_file = '/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/MATLAB structure/EEGMMIDB_Curated.mat'
    read_annotations(eeg_file)
