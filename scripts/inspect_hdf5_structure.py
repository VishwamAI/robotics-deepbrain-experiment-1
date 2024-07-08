import h5py

def print_hdf5_structure(file_path):
    """
    Prints the structure of an HDF5 file.

    Args:
    file_path (str): Path to the HDF5 file.

    Returns:
    None
    """
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            print(key)
            if key == 'EEGMMIDB':
                for sub_key in f[key].keys():
                    print(f"    {sub_key}")

if __name__ == "__main__":
    hdf5_file_path = '/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/MATLAB structure/EEGMMIDB_Curated.mat'
    print_hdf5_structure(hdf5_file_path)
