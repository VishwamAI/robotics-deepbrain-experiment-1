import pandas as pd
import os

def process_annotations(annotation_dir, output_file):
    """
    Process annotation files to generate meaningful labels for EEG data.

    Args:
    annotation_dir (str): Directory containing annotation files.
    output_file (str): Path to save the processed labels.

    Returns:
    None
    """
    labels = []

    # Check if the annotation directory exists
    if not os.path.exists(annotation_dir):
        print(f"Error: The directory {annotation_dir} does not exist.")
        return

    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Print the annotation directory path
    print(f"Annotation directory: {annotation_dir}")

    # Iterate over annotation files in the directory
    annotation_files_found = False
    for filename in os.listdir(annotation_dir):
        full_path = os.path.join(annotation_dir, filename)
        print(f"Checking file: {full_path}")
        if "_ann" in filename.lower() and filename.lower().endswith(".csv"):
            annotation_files_found = True
            file_path = os.path.join(annotation_dir, filename)
            ann_df = pd.read_csv(file_path, header=None)

            # Extract labels from the annotation file
            for _, row in ann_df.iterrows():
                event_type = row[0]
                start_sample = int(row[3])
                end_sample = int(row[4])
                labels.extend([event_type] * (end_sample - start_sample + 1))

    if not annotation_files_found:
        print(f"Error: No annotation files ending with '_ann.csv' found in the directory {annotation_dir}.")
        return

    # Save the labels to the output file
    labels_df = pd.DataFrame(labels, columns=["Label"])
    labels_df.to_csv(output_file, index=False)
    print(f"Labels processed and saved to {output_file}")

if __name__ == "__main__":
    annotation_dir = os.path.abspath('/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files')
    output_file = 'processed_labels.csv'
    process_annotations(annotation_dir, output_file)
