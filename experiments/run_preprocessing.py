from src.data_preprocessing import create_imbalance_datasets, convert_original_dataset
from src.utils import create_folder, get_pre_data_augmentation_folder_name

# Runs preprocessing on a dataset
def run_preprocessing(dataset):
    create_folder(dataset['dataset_folder'], get_pre_data_augmentation_folder_name())

    # Convert the original dataset categorical columns    
    folder_name, encoded_file_name, minority_class_value = convert_original_dataset(
        dataset_folder=dataset['dataset_folder'],
        dataset_original_file=dataset['dataset_original_file'],
        categorical_columns=dataset['categorical_columns'],
        target_column=dataset['target_column'],
        minority_class=dataset['minority_class']
    )

    # If the dataset is originally balanced, create synthetic imbalances
    if dataset['balanced']: 
        # Creates the imbalanced datasets
        imbalanced_file_location = create_imbalance_datasets(
            dataset_folder=dataset['dataset_folder'],
            dataset_original_file=encoded_file_name,
            target_column=dataset['target_column'],
            minority_class=minority_class_value,
            imbalance_ratios=dataset['imbalance_ratios'],
            categorical_columns=dataset['categorical_columns']
        )
    else:
        imbalanced_file_location = None
    return encoded_file_name, imbalanced_file_location, minority_class_value
