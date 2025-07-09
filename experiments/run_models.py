from src.model import create_model


def run_models(file_locations, target_column):
    trained_models = []
    for file_location in file_locations:
        trained_model, testing_data = create_model(file_location, target_column)
        trained_models.append((file_location, trained_model))    
    return trained_models, testing_data