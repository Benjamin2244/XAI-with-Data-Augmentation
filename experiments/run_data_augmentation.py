from src.data_augmentation import create_data_augmentation

### Use data augmnentation to create balanced datasets
def run_data_augmentation(imbalanced_file_locations, target_column):
    all_balanced_file_locations = []
    for file_location in imbalanced_file_locations:
        print(f"all imbalanced file locations: {imbalanced_file_locations}")
        balanced_file_location = create_data_augmentation(file_location[0], file_location[1], target_column)
        all_balanced_file_locations.append(balanced_file_location)
    return all_balanced_file_locations
