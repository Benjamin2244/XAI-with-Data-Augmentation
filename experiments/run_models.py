from src.model import create_model

def run_model(file_location, target_column):
    trained_model, training_data, testing_data = create_model(file_location, target_column)
    return trained_model, training_data, testing_data

def run_models(file_locations, target_column):
    trained_models = []
    for file_location in file_locations:
        trained_model, training_data, testing_data = run_model(file_location, target_column)
        trained_models.append((file_location[1], trained_model, training_data, testing_data))    
    return trained_models