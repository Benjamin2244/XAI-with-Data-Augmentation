from src.data_augmentation import create_data_augmentation
from src.utils import create_folder, create_da_folder, get_data_augmentation_folder_name, get_SMOTE_folder_name

### Use data augmnentation to create balanced datasets
def run_data_augmentation(imbalanced_file_locations, target_column):
    create_folder(imbalanced_file_locations[0][0], get_data_augmentation_folder_name())
    create_da_folder(imbalanced_file_locations[0][0], get_SMOTE_folder_name())
    all_balanced_file_locations = []
    for file_location in imbalanced_file_locations:
        print(f"all imbalanced file locations: {imbalanced_file_locations}")
        balanced_file_location = create_data_augmentation(file_location[0], file_location[1], target_column)
        all_balanced_file_locations.append(balanced_file_location)

    return all_balanced_file_locations
