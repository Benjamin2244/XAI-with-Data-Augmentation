from .utils import get_parent_directory, read_csv_file, get_pre_data_augmentation_folder_name, get_data_augmentation_folder_name, get_SMOTE_folder_name, get_GAN_folder_name
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
import pandas as pd

def apply_smote(df, target_column):
    smote = SMOTE(random_state=24)

    X = df.drop(columns=[target_column]).values
    y = df[target_column]

    X, y = smote.fit_resample(X, y)

    df_smote = pd.DataFrame(X, columns=df.columns[:-1])
    df_smote['target'] = y
    return df_smote

# Save the balanced dataset
def save_balanced_dataset(df, dataset_folder, data_augmentation_type, dataset_new_file):
    parent_dir = get_parent_directory()
    output_file = parent_dir / 'data' / dataset_folder / get_data_augmentation_folder_name() / data_augmentation_type / dataset_new_file
    df.to_csv(output_file, index=False)

def get_new_file_name_SMOTE(dataset_imbalanced_file_name):
    return dataset_imbalanced_file_name.replace(".csv", "_with_smote.csv")

def get_new_file_name_GAN(dataset_imbalanced_file_name):
    return dataset_imbalanced_file_name.replace(".csv", "_with_GAN.csv")

# Run data augmentation
def create_data_augmentation_SMOTE(dataset_folder, dataset_imbalanced_file_name, target_column):
    df = read_csv_file(dataset_folder, get_pre_data_augmentation_folder_name(), dataset_imbalanced_file_name)
    df_smote = apply_smote(df, target_column)
    new_file_name = get_new_file_name_SMOTE(dataset_imbalanced_file_name)
    save_balanced_dataset(df_smote, dataset_folder, get_SMOTE_folder_name(), new_file_name)
    return (dataset_folder, new_file_name)

def apply_GAN(df, categorical_columns, num_required_rows):
    ctgan = CTGAN(epochs=10)
    ctgan.fit(df, categorical_columns)
    da_dataset = ctgan.sample(num_required_rows)
    return da_dataset

# Run data augmentation
# Trains GAN on minority class only
# Sample is made and added to df
def create_data_augmentation_GAN(dataset_folder, dataset_imbalanced_file_name, target_column, minority_class, categorical_columns):
    df = read_csv_file(dataset_folder, get_pre_data_augmentation_folder_name(), dataset_imbalanced_file_name)
    
    mask = df[target_column] == minority_class
    df_minority = df[mask]

    class_counts = df[target_column].value_counts()
    num_max_class = class_counts.max()
    num_minority = len(df_minority)
    num_required_rows = num_max_class - num_minority

    df_GAN = apply_GAN(df_minority, categorical_columns, num_required_rows)
    
    balanced_df = pd.concat([df, df_GAN], ignore_index=True)
    
    new_file_name = get_new_file_name_GAN(dataset_imbalanced_file_name)
    save_balanced_dataset(balanced_df, dataset_folder, get_GAN_folder_name(), new_file_name)

    return (dataset_folder, new_file_name)