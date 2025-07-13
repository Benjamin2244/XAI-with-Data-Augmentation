from pathlib import Path
import pandas as pd
import os

# Check if a file exists
def file_exists(file_path):
    file = Path(file_path)
    return file.exists()

# Get all files in a directory
def get_files_in_directory(directory):
    dir_path = Path(directory)
    return [file for file in dir_path.iterdir() if file.is_file()]

# Get file path
def get_file_path(folder_purpose, folder, file_name):
    parent_dir = get_parent_directory()
    return parent_dir / folder_purpose / folder / file_name

# Get parent directory of the current file
def get_parent_directory():
    parent_dir = Path(__file__).parent
    return parent_dir.parent

# Read the CSV file
def read_csv_file(*path_parts):
    parent_dir = get_parent_directory()
    csv_file = parent_dir / 'data' / Path(*path_parts)
    if not csv_file.exists():
        raise FileNotFoundError(f"File {csv_file} does not exist.")
    return pd.read_csv(csv_file)

# Checks if a folder exists
def folder_exists(folder_path):
    folder = Path(folder_path)
    return folder.exists() and folder.is_dir()

# Creates folder
def create_folder(dataset_name, folder_name):
    parent_dir = get_parent_directory()
    folder_path = parent_dir / 'data' / dataset_name / folder_name
    if folder_path.exists() and folder_path.is_dir():
        pass # Folder exists
    else:
        folder_path.mkdir(parents=True, exist_ok=True) # Creates the folder

# Creates folder
def create_da_folder(dataset_name, da_type):
    parent_dir = get_parent_directory()
    folder_path = parent_dir / 'data' / dataset_name / get_data_augmentation_folder_name() / da_type
    if folder_path.exists() and folder_path.is_dir():
        pass # Folder exists
    else:
        folder_path.mkdir(parents=True, exist_ok=True) # Creates the folder

# Prints a progress dot
def print_progress_dot():
    print('.', end='', flush=True)

# Prints a progress dot for optuna trials
def print_progress_dot_optuna(study, trial):
    print_progress_dot()

def get_pre_data_augmentation_folder_name():
    return 'Pre Data Augmentation'

def get_data_augmentation_folder_name():
    return 'Data Augmentation'

def get_SMOTE_folder_name():
    return 'SMOTE'
