from src.data_augmentation import create_data_augmentation, get_new_file_name
from src.utils import create_folder, create_da_folder, get_data_augmentation_folder_name, get_SMOTE_folder_name,  get_file_path, file_exists

### Use data augmnentation to create balanced datasets
def run_data_augmentation(imbalanced_file_locations, target_column):
    create_folder(imbalanced_file_locations[0][0], get_data_augmentation_folder_name())
    create_da_folder(imbalanced_file_locations[0][0], get_SMOTE_folder_name())
    all_balanced_file_locations = []
    for file_location in imbalanced_file_locations:
        new_file_name = get_new_file_name(file_location[1])
        new_file_path = get_file_path(file_location[0], get_data_augmentation_folder_name(), get_SMOTE_folder_name(), new_file_name)
        if file_exists(new_file_path):
            print(f"File {new_file_name} already exists. Skipping conversion.")
            balanced_file_location = (file_location[0], new_file_name)
        else:
            balanced_file_location = create_data_augmentation(file_location[0], file_location[1], target_column)
        all_balanced_file_locations.append(balanced_file_location)

    return all_balanced_file_locations
