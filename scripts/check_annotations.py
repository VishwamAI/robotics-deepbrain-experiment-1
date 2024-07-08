import os

def check_annotations(annotation_dir):
    annotation_files_found = False
    total_files = 0
    annotation_files_count = 0
    print('Starting to check files...')
    for filename in os.listdir(annotation_dir):
        total_files += 1
        full_path = os.path.join(annotation_dir, filename)
        print(f'Checking file: {full_path}')
        if '_ann.csv' in filename.lower():
            annotation_files_found = True
            annotation_files_count += 1
            print(f'Annotation file found: {filename}')
        else:
            print(f'Not an annotation file: {filename}')
    print(f'Total files checked: {total_files}')
    print(f'Annotation files detected: {annotation_files_count}')
    if annotation_files_found:
        print('Annotation files found: True')
    else:
        print('Annotation files found: False')

if __name__ == "__main__":
    annotation_dir = os.path.abspath('/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files')
    check_annotations(annotation_dir)
