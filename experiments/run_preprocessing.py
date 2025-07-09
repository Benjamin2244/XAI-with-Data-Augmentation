from src.data_preprocessing import create_imbalance_datasets, convert_original_dataset


def run_preprocessing(dataset):
    print("Running preprocessing...")
    folder_name, encoded_file_name = convert_original_dataset(
        dataset_folder=dataset['dataset_folder'],
        dataset_original_file=dataset['dataset_original_file'],
        categorical_columns=dataset['categorical_columns']
    )
    imbalanced_file_location = create_imbalance_datasets(
        dataset_folder=dataset['dataset_folder'],
        dataset_original_file=encoded_file_name,
        target_column=dataset['target_column'],
        minority_class=dataset['minority_class'],
        imbalance_ratios=dataset['imbalance_ratios'],
        categorical_columns=dataset['categorical_columns']
    )
    return encoded_file_name, imbalanced_file_location
