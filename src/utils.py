from pathlib import Path
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def force_pt_extension(file_name):
    if file_name.endswith('.pt'):
        return file_name
    if '.' in file_name:
        file_name = file_name.rsplit('.', 1)[0]
    return file_name + '.pt'

def force_csv_extension(file_name):
    if file_name.endswith('.csv'):
        return file_name
    if '.' in file_name:
        file_name = file_name.rsplit('.', 1)[0]
    return file_name + '.csv'

# Check if a file exists
def file_exists(file_path):
    file = Path(file_path)
    return file.exists()

# Get all files in a directory
def get_files_in_directory(directory):
    dir_path = Path(directory)
    return [file for file in dir_path.iterdir() if file.is_file()]

# Get file path
def get_file_path(*path_parts):
    parent_dir = get_parent_directory()
    return parent_dir / 'data' / Path(*path_parts)

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

# Reset the folder
def reset_folder(dataset_name, folder_name):
    parent_dir = get_parent_directory()
    folder_name = Path(folder_name)
    folder_path = parent_dir / 'data' / dataset_name / folder_name
    if folder_path.exists() and folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file() or file.is_symlink():
                print(f"Deleting file: {file}")
                file.unlink()
            elif file.is_dir():
                reset_folder(dataset_name, folder_name / file.name)
                
# Delete test dataset
def delete_test_dataset(dataset_folder):
    parent_dir = get_parent_directory()
    folder_path = parent_dir / 'data' / dataset_folder 
    if folder_path.exists():
        for file in folder_path.iterdir():
            if file.is_file() and file.name.startswith('test_'):
                print(f"Deleting file: {file}")
                file.unlink()

def is_control(name):
    if name.startswith("encoded_"):
        return True
    return False

def is_SMOTE(name):
    if "_smote" in name:
        return True
    return False

def is_GAN(name):
    if "_GAN" in name:
        return True
    return False

# Load columnn names
def load_column_names(dataset_folder, target_column):
    parent_dir = get_parent_directory()
    folder_path = parent_dir / 'data' / dataset_folder

    for file in folder_path.iterdir():
        if file.is_file() and file.name.startswith('test_'):
            test_file = file.name
            break

    if not test_file:
        raise FileNotFoundError(f"No test file found in {dataset_folder}.")
    
    df = read_csv_file(dataset_folder, test_file)

    feature_names = df.drop(columns=[target_column]).columns.tolist()

    return feature_names

# Load test dataset
def load_test_data(dataset_folder, target_column):
    parent_dir = get_parent_directory()
    folder_path = parent_dir / 'data' / dataset_folder

    for file in folder_path.iterdir():
        if file.is_file() and file.name.startswith('test_'):
            test_file = file.name
            break

    if not test_file:
        raise FileNotFoundError(f"No test file found in {dataset_folder}.")
    
    df = read_csv_file(dataset_folder, test_file)

    X_test = df.drop(columns=[target_column]).astype(float).values
    y_test = df[target_column].values

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    testing_data = X_test, y_test
    return testing_data


class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
# def predict(model, x):
#     model.eval()
#     with torch.no_grad():
#         if hasattr(x, "to_numpy"):
#             x = x.to_numpy()
#         if not isinstance(x, torch.Tensor):
#             x = torch.tensor(x, dtype=torch.float32)
#         prediction = model(x)
#         return prediction.numpy()

def predict(model, x):
    model.eval()
    with torch.no_grad():
        if hasattr(x, "to_numpy"):
            x = x.to_numpy()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        prediction = model(x)

        if prediction.shape[1] == 1: # Checks if binary classification
            positive_probability = torch.sigmoid(prediction)
            probability = torch.cat([1 - positive_probability, positive_probability], dim=1)
        else: 
            probability = F.softmax(prediction, dim=1)

        return prediction.numpy()

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=24)

    X_train = X_train.astype(float)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    X_test = X_test.astype(float)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    return X_train, X_test, y_train, y_test

def load_dataset(file_location, dataset_type, da_subfolder=None):
    dataset_folder, dataset_file_name = file_location
    dataset_file_name = force_csv_extension(dataset_file_name)

    if dataset_type == 'pre':
        path = [dataset_folder, get_pre_data_augmentation_folder_name(), dataset_file_name]
    elif dataset_type == 'da':
        path = [dataset_folder, get_data_augmentation_folder_name(), da_subfolder, dataset_file_name]

    df = read_csv_file(*path)
    return df

def get_num_features(file_location, target_column, dataset_type, da_subfolder):
    df = load_dataset(file_location, dataset_type, da_subfolder) # Load dataset
    X_train, X_val, y_train, y_val = split_data(df, target_column) # Split data
    num_features = X_train.shape[1] # Train a model on the dataset
    return num_features

def get_num_classes(file_location, target_column, dataset_type, da_subfolder):
    df = load_dataset(file_location, dataset_type, da_subfolder) # Load dataset
    X_train, X_val, y_train, y_val = split_data(df, target_column) # Split data
    num_classes = len(torch.unique(y_train))
    return num_classes

def load_model(file_location, target_column, dataset_type, da_subfolder):
    dataset_folder, dataset_name = file_location
    dataset_name = force_pt_extension(dataset_name)
    parent_dir = get_parent_directory()
    path = parent_dir / 'data' / dataset_folder / get_model_folder_name() / f"{dataset_name}"

    num_features = get_num_features(file_location, target_column, dataset_type, da_subfolder)
    num_classes = get_num_classes(file_location, target_column, dataset_type, da_subfolder)

    model = NeuralNetwork(num_features=num_features, num_classes=num_classes)  # Dummy model to load state dict
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model

def get_encoded_categorical_columns(dataset_folder, encoded_file_name):
    df = load_dataset((dataset_folder, encoded_file_name), 'pre')
    categorical_columns = []

    for col in df.columns:
        isBinary = True
        isBool = True
        for row in df[col]:
            if (isBinary == False) and (isBool == False):
                break 
            if row not in {0, 1}:
                isBinary = False
            if row not in {True, False}:
                isBool = False
        if (isBinary or isBool):
            categorical_columns.append(col)
    return categorical_columns

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

def get_GAN_folder_name():
    return 'GAN'

def get_model_folder_name():
    return 'Models'
