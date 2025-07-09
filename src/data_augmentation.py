from .utils import get_parent_directory, read_csv_file
from imblearn.over_sampling import SMOTE
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
def save_balanced_dataset(df, dataset_folder, dataset_new_file):
    parent_dir = get_parent_directory()
    output_file = parent_dir / 'data' / dataset_folder / dataset_new_file
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")


# Run data augmentation
def create_data_augmentation(dataset_folder, dataset_imbalanced_file_name, target_column):
    print("============= Running data augmentation =============")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Dataset imbalanced file: {dataset_imbalanced_file_name}")
    df = read_csv_file(dataset_folder, dataset_imbalanced_file_name)
    df_smote = apply_smote(df, target_column)
    new_file_name = dataset_imbalanced_file_name.replace(".csv", "_with_smote.csv")
    save_balanced_dataset(df_smote, dataset_folder, new_file_name)
    return (dataset_folder, new_file_name)