import os

def read_h5ls_output_in_chunks(file_path, chunk_size=1024):
    """
    Reads the output file from h5ls command in chunks and searches for lines containing 'Annotations'.

    Args:
    file_path (str): Path to the output file from h5ls command.
    chunk_size (int): Size of each chunk to read from the file.

    Returns:
    list: List of lines containing 'Annotations'.
    """
    annotations_keys = []
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Listing files in output directory: {os.listdir('output')}")
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return []
        with open(file_path, 'r') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                lines = chunk.split('\n')
                for line in lines:
                    if 'Annotations' in line:
                        annotations_keys.append(line.strip())
        return annotations_keys
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    output_file_path = 'output/h5ls_r_Physionet_EEG_1720429776.6928756.txt'
    keys = read_h5ls_output_in_chunks(output_file_path)
    for key in keys:
        print(key)
