from .run_preprocessing import run_preprocessing
from .run_data_augmentation import run_data_augmentation
from .run_models import run_models


if __name__ == "__main__":
    heart_failure_data = {
        'dataset_folder': 'heart_failure_prediction',
        'dataset_original_file': 'heart.csv',
        'target_column': 'HeartDisease',
        'minority_class': 1,
        'imbalance_ratios': [0.1, 0.5, 1.0],
        'categorical_columns': ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
    }
    datasets = [heart_failure_data]
    for dataset in datasets:
        print(f"Runnning preprocessing for dataset: {dataset['dataset_folder']}")
        imbalanced_file_locations = run_preprocessing(dataset)
        print(f"Running data augmentation for dataset: {dataset['dataset_folder']}")
        da_file_locations = run_data_augmentation(imbalanced_file_locations, dataset['target_column'])
        ### Train models
        # Train original
        # Train each imbalanced dataset
        # Train each balanced dataset
        print(f"Running models for dataset: {dataset['dataset_folder']}")
        models, testing_data = run_models(da_file_locations, dataset['target_column'])
        ### Evaluate models
        # F-1 score
        
        # SHAP comparisons
    print("All experiments completed.")

