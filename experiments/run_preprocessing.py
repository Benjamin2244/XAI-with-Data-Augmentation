from src.data_preprocessing import create_imbalance_datasets


def run_preprocessing(dataset):
    print("Running preprocessing...")
    imbalanced_file_location = create_imbalance_datasets(
        dataset_folder=dataset['dataset_folder'],
        dataset_original_file=dataset['dataset_original_file'],
        target_column=dataset['target_column'],
        minority_class=dataset['minority_class'],
        imbalance_ratios=dataset['imbalance_ratios'],
        categorical_columns=dataset['categorical_columns']
    )
    return imbalanced_file_location
