def find_annotations_keys(file_path, output_path):
    """
    Reads the output file from h5ls command and searches for lines containing 'Annotations'.

    Args:
    file_path (str): Path to the output file from h5ls command.
    output_path (str): Path to the output file where found keys will be written.

    Returns:
    list: List of lines containing 'Annotations'.
    """
    annotations_keys = []
    with open(file_path, 'r') as file:
        with open(output_path, 'w') as output_file:
            for line in file:
                if 'Annotations' in line:
                    annotations_keys.append(line.strip())
                    output_file.write(f"Found annotation key: {line.strip()}\n")  # Write to output file
    return annotations_keys

if __name__ == "__main__":
    output_file_path = '/home/ubuntu/output/h5ls_r_Physionet_EEG_1720429776.6928756.txt'
    result_file_path = '/home/ubuntu/output/annotations_keys_output.txt'
    print(f"Output file path: {output_file_path}")  # Diagnostic print statement
    keys = find_annotations_keys(output_file_path, result_file_path)
    for key in keys:
        print(key)
