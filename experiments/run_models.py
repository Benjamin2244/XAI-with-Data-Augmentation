from src.model import create_model, does_model_exist
from src.utils import create_folder, get_model_folder_name, load_model

def run_model(file_location, target_column, dataset_type, da_subfolder):
    dataset_folder, dataset_name = file_location

    create_folder(dataset_folder, get_model_folder_name())

    model_file = f"{dataset_name.removesuffix('.csv')}.pt"
    if does_model_exist((dataset_folder, model_file)):
        model = load_model((dataset_folder, model_file), target_column, dataset_type, da_subfolder)
        print(f"Model for {model_file} already exists. Loading the model.")
    else:
        print(f"Model for {model_file} does not exist. Creating a new model.")
        model = create_model(file_location, target_column, dataset_type, da_subfolder)
    return model

def run_models(file_locations, target_column, dataset_type, da_subfolder):
    trained_models = []
    for file_location in file_locations:
        trained_model = run_model(file_location, target_column, dataset_type, da_subfolder)
        trained_models.append((file_location[1], trained_model))    
    return trained_models