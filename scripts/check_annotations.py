import os

def check_annotations(annotation_dir):
    annotation_files_found = False
    print('Starting to check files...')
    for filename in os.listdir(annotation_dir):
        full_path = os.path.join(annotation_dir, filename)
        print(f'Checking file: {full_path}')
        if filename.lower().endswith('_ann.csv'):
            annotation_files_found = True
            print(f'Annotation file found: {filename}')
    if annotation_files_found:
        print('Annotation files found: True')
    else:
        print('Annotation files found: False')

if __name__ == "__main__":
    annotation_dir = os.path.abspath('/home/ubuntu/robotics-deepbrain-experiment-1/Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/CSV files')
    check_annotations(annotation_dir)
