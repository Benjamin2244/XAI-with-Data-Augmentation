from .run_preprocessing import run_preprocessing
from .run_data_augmentation import run_data_augmentation
from .run_models import run_models, run_model
from .run_analysis import all_analysis
import random
import numpy as np
import torch
import time


def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def run_experiments_full(datasets):
    for dataset in datasets:
        models = {}
        print(f"Runnning preprocessing for dataset: {dataset['dataset_folder']}")
        encoded_file_name, imbalanced_file_locations = run_preprocessing(dataset)

        print(f"Running data augmentation for dataset: {dataset['dataset_folder']}")
        da_file_locations = run_data_augmentation(imbalanced_file_locations, dataset['target_column'])

        continue

        print(f"Training models for dataset: {dataset['dataset_folder']}")
        model, training_data, testing_data = run_model((dataset['dataset_folder'], encoded_file_name), dataset['target_column'])
        models[encoded_file_name] = {'model': model, 'training_data':training_data, 'testing_data': testing_data}

        da_models = run_models(da_file_locations, dataset['target_column'])
        for da_model in da_models:
            file_location, model, training_data, testing_data = da_model
            print(f"Training model for augmented data: {file_location}")
            models[file_location] = {'model': model, 'training_data':training_data, 'testing_data': testing_data}

        print(f"Evaluating dataset: {dataset['dataset_folder']}")
        all_analysis(models)
        # F-1 score
        # SHAP comparisons
    print("All experiments completed.")

if __name__ == "__main__":
    set_seed(24)
    heart_failure_data = {
        'dataset_folder': 'heart_failure_prediction',
        'dataset_original_file': 'heart.csv',
        'target_column': 'HeartDisease',
        'minority_class': 1,
        'imbalance_ratios': [0.1, 0.5, 1.0],
        'categorical_columns': ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
    }
    datasets = [heart_failure_data]

    print("Starting experiments...")
    start_time = time.time()

    run_experiments_full(datasets)

    end_time = time.time()
    print(f"Experiments completed in {end_time - start_time:.2f} seconds.")