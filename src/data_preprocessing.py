from .utils import get_parent_directory, read_csv_file, file_exists, get_file_path, get_pre_data_augmentation_folder_name
import pandas as pd
from sklearn.model_selection import train_test_split


# Create an imbalance in the dataset
def create_imbalance(df, target_column, minority_class, imbalance_ratio=0.1):
    imbalanced_df = df.copy()
    classes = imbalanced_df[target_column].unique()
    class_data = {}

    dataset_size = len(imbalanced_df)
    minority_size = len(imbalanced_df[imbalanced_df[target_column] == minority_class])
    non_minority_class_size = dataset_size - minority_size

    for class_type in classes:
        class_data[class_type] = imbalanced_df[imbalanced_df[target_column] == class_type]
    new_minority_size = int(non_minority_class_size * imbalance_ratio)

    # Randomly remove samples from the minority class to create imbalance
    if new_minority_size <= minority_size: 
        sampled_minority = class_data[minority_class].sample(n=new_minority_size, random_state=24)
        imbalanced_df = pd.concat([sampled_minority] + [class_data[c] for c in classes if c != minority_class])
    # Randomly remove samples from the non-minority class to create imbalance
    else: 
        new_non_minority_size = int(minority_size / imbalance_ratio)
        non_minority_data = pd.concat([class_data[c] for c in classes if c != minority_class])
        sampled_non_minority = non_minority_data.sample(n=new_non_minority_size, random_state=24)
        imbalanced_df = pd.concat([class_data[minority_class]] + [sampled_non_minority])

    return imbalanced_df

# Convert categorical columns to category type
def convert_categorical_columns(df_imbalance, categorical_columns):
    df_encoded = pd.get_dummies(df_imbalance, columns=categorical_columns)
    return df_encoded

# Save the dataset
def save_dataset(df, dataset_folder, dataset_new_file):
    parent_dir = get_parent_directory()
    output_file = parent_dir / 'data' / dataset_folder / get_pre_data_augmentation_folder_name() / dataset_new_file
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

def save_test_dataset(df, dataset_folder, dataset_new_file):
    parent_dir = get_parent_directory()
    output_file = parent_dir / 'data' / dataset_folder / dataset_new_file
    df.to_csv(output_file, index=False)
    print(f"Test dataset saved to {output_file}")

# Create the imbalance datasets
def create_imbalance_datasets(dataset_folder, dataset_original_file, target_column, minority_class, imbalance_ratios, categorical_columns):
    df = read_csv_file(dataset_folder, get_pre_data_augmentation_folder_name(), dataset_original_file)
    imbalanced_file_locations = []
    for imbalance_ratio in imbalance_ratios:
        if dataset_original_file.startswith("encoded_"):
            dataset_original_file = dataset_original_file.replace("encoded_", "")
        new_file_name = f"imbalanced_{imbalance_ratio}_{dataset_original_file}"

        file_path = get_file_path(dataset_folder, get_pre_data_augmentation_folder_name(), new_file_name)

        if file_exists(file_path):
            print(f"File {new_file_name} already exists. Skipping conversion.")
        else:
            df_imbalance = create_imbalance(df, target_column, minority_class, imbalance_ratio)
            save_dataset(df_imbalance, dataset_folder, new_file_name)
        imbalanced_file_locations.append((dataset_folder, new_file_name))
    return imbalanced_file_locations

def df_train_test_split(df, target_column, test_size=0.2):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=24, stratify=df[target_column])
    return df_train, df_test

# Convert the original dataset categorical columns 
def convert_original_dataset(dataset_folder, dataset_original_file, categorical_columns, target_column):
    new_file_name = f"encoded_{dataset_original_file}"

    file_path = get_file_path(dataset_folder, get_pre_data_augmentation_folder_name(), new_file_name)

    if file_exists(file_path):
        print(f"File {new_file_name} already exists. Skipping conversion.")
        return dataset_folder, new_file_name

    df = read_csv_file(dataset_folder, dataset_original_file)
    df_encoded = convert_categorical_columns(df, categorical_columns)

    training_df, testing_df= df_train_test_split(df_encoded, target_column)

    save_dataset(training_df, dataset_folder, new_file_name)
    save_test_dataset(testing_df, dataset_folder, f"test_{dataset_original_file}")
    return dataset_folder, new_file_name