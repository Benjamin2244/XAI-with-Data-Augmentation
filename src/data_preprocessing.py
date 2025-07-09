from .utils import get_parent_directory, read_csv_file
import pandas as pd


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
    output_file = parent_dir / 'data' / dataset_folder / dataset_new_file
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

# Create the imbalance datasets
def create_imbalance_datasets(dataset_folder, dataset_original_file, target_column, minority_class, imbalance_ratios, categorical_columns):
    df = read_csv_file(dataset_folder, dataset_original_file)
    imbalanced_file_location = []
    for imbalance_ratio in imbalance_ratios:
        df_imbalance = create_imbalance(df, target_column, minority_class, imbalance_ratio)
        df_imbalance = convert_categorical_columns(df_imbalance, categorical_columns)
        new_file_name = f"imbalanced_{imbalance_ratio}_{dataset_original_file}"
        save_dataset(df_imbalance, dataset_folder, new_file_name)
        imbalanced_file_location.append((dataset_folder, new_file_name))
    return imbalanced_file_location