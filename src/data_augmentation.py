from pathlib import Path
import pandas as pd

import pandas as pd

# Get parent directory of the current file
def get_parent_directory():
    parent_dir = Path(__file__).parent
    return parent_dir.parent

# Read the CSV file
def read_csv_file(dataset_folder, dataset_original_file):
    parent_dir = get_parent_directory()
    csv_file = parent_dir / 'data' / dataset_folder / dataset_original_file
    if not csv_file.exists():
        raise FileNotFoundError(f"File {csv_file} does not exist.")
    return pd.read_csv(csv_file)

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

# Save the dataset
def save_dataset(df, dataset_folder, dataset_new_file):
    parent_dir = get_parent_directory()
    output_file = parent_dir / 'data' / dataset_folder / dataset_new_file
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

# Create the imbalance datasets
def create_imbalance_datasets(dataset_folder, dataset_original_file, target_column, minority_class, imbalance_ratios):
    df = read_csv_file(dataset_folder, dataset_original_file)
    for imbalance_ratio in imbalance_ratios:
        df_imbalance = create_imbalance(df, target_column, minority_class, imbalance_ratio)
        save_dataset(df_imbalance, dataset_folder, f"imbalanced_{imbalance_ratio}_{dataset_original_file}")

if __name__ == "__main__":
    create_imbalance_datasets(
        dataset_folder='heart_failure_prediction',
        dataset_original_file='heart.csv',
        target_column='HeartDisease',
        minority_class=1,
        imbalance_ratios=[0.1, 0.5, 1.0]
    )