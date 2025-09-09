from .run_preprocessing import run_preprocessing
from .run_data_augmentation import run_data_augmentation
from .run_models import run_models, run_model
from .run_analysis import all_analysis
from src.utils import reset_folder, get_pre_data_augmentation_folder_name, get_data_augmentation_folder_name, delete_test_dataset, get_model_folder_name, get_encoded_categorical_columns, get_results_folder_name
import random
import numpy as np
import torch
import time

# Sets seeds
def set_seed(seed=24):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Reset the contents of each folder for each dataset
# Only the original dataset should remain
def reset_folders(datasets):
    for dataset in datasets:
        dataset_folder = dataset['dataset_folder']
        reset_folder(dataset_folder, get_pre_data_augmentation_folder_name())
        reset_folder(dataset_folder, get_data_augmentation_folder_name())
        reset_folder(dataset_folder, get_model_folder_name())
        reset_folder(dataset_folder, get_model_folder_name())
        delete_test_dataset(dataset_folder, get_results_folder_name())
        
# Highest level of the data pipeline
def run_experiments_full(datasets):
    ### Uncomment to delete all progress on experiments ### 
    # reset_folders(datasets)
    # continue
    ### Think carefully ###
    dataset_models = {}
    # Goes through one dataset at a time
    for dataset in datasets: 
        models = {}

        dataset_folder = dataset['dataset_folder']

        print(f"Runnning preprocessing for dataset: {dataset_folder}")
        encoded_file_name, imbalanced_file_locations, minority_class_value = run_preprocessing(dataset)

        dataset['minority_class'] = minority_class_value
        dataset['encoded_categorical_columns'] = get_encoded_categorical_columns(dataset_folder, encoded_file_name)

        print(f"Running data augmentation for dataset: {dataset_folder}")
        if imbalanced_file_locations == None:
            da_file_locations = run_data_augmentation([(dataset_folder, encoded_file_name)], dataset['target_column'], dataset['minority_class'], dataset['encoded_categorical_columns']) 
        else:
            da_file_locations = run_data_augmentation(imbalanced_file_locations, dataset['target_column'], dataset['minority_class'], dataset['encoded_categorical_columns']) 
        
        print(f"Training models for dataset: {dataset_folder}")
        dataset_type, da_subfolder = 'pre', None
        model = run_model((dataset_folder, encoded_file_name), dataset['target_column'], dataset_type, da_subfolder)
        models[encoded_file_name] = {'dataset': dataset_folder, 'model': model, 'target_column': dataset['target_column'], 'minority_class': dataset['minority_class']}
    
        da_models = run_models(da_file_locations, dataset['target_column'])
        for da_model in da_models:
            file_location, model = da_model
            models[file_location] = {'dataset': dataset_folder, 'model': model, 'target_column': dataset['target_column'], 'minority_class': dataset['minority_class']}

        dataset_models[dataset['dataset_folder']] = models

    for dataset in datasets:
        print(f"Evaluating dataset: {dataset['dataset_folder']}")
        all_analysis(dataset_models[dataset['dataset_folder']], dataset['dataset_folder'])

# Entry point for the experiments
if __name__ == "__main__":
    set_seed(24) # Sets the seeds

    example_data = {
        'dataset_folder': 'folder_name', # Name of the folder where the data lies
        'dataset_original_file': 'dataset.csv', # Name of the dataset inside the folder
        'target_column': 'Target', # Name of the column to predict
        'minority_class': 'True', # Name of the class that we want to act as the minority
        'imbalance_ratios': [0.1, 0.5, 1.0], # Different ratios for the minority data to take e.g. 0.1 is 1:10 and 0.5 is 5:10
        'categorical_columns': ['Category A', 'Category B'], # List of what columns have categorical data (DO NOT ADD THE TARGET COLUMN)
        'balanced': True, # True if the original dataset is some what balanced, and synthetic imbalances should be created
    }

    heart_failure_data = {
        'dataset_folder': 'heart_failure_prediction',
        'dataset_original_file': 'heart.csv',
        'target_column': 'HeartDisease',
        'minority_class': 1, # (1 = 508) (0 = 410)
        'imbalance_ratios': [0.1, 0.5, 1.0],
        'categorical_columns': ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'],
        'balanced': True,
    }

    personality_data = {
        'dataset_folder': 'personality',
        'dataset_original_file': 'personality_dataset.csv',
        'target_column': 'Personality',
        'minority_class': 'Introvert', # (Introvert = 1,409) (Extrovert = 1,491)
        'imbalance_ratios': [0.1, 0.5, 1.0],
        'categorical_columns': [],
        'balanced': True,
    }

    forest_cover_data = {
        'dataset_folder': 'forest_cover',
        'dataset_original_file': 'covtype.csv',
        'target_column': 'Cover_Type',
        'minority_class': 4, # (1 = 211,840) (2 = 283,301) (3 = 35,754) (4 = 2,747) (5 = 9,493) (6 = 17,367) (7 = 20,510)
        'imbalance_ratios': [0.1, 0.5, 1.0],
        'categorical_columns': [],
        'balanced': False, # False = Ignores 'imbalance_ratios'
    }
    datasets = [heart_failure_data, personality_data, forest_cover_data]
    # datasets = [forest_cover_data]
    # datasets = [heart_failure_data]
    # datasets = [personality_data]
    # datasets = [heart_failure_data, personality_data]

    print("Starting experiments...")
    start_time = time.time()

    run_experiments_full(datasets)

    end_time = time.time()
    print(f"Experiments completed in {end_time - start_time:.2f} seconds.")