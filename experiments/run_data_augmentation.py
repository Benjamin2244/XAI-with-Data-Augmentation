from src.data_augmentation import create_data_augmentation_SMOTE, create_data_augmentation_GAN, get_new_file_name_SMOTE, get_new_file_name_GAN
from src.utils import create_folder, create_da_folder, get_data_augmentation_folder_name, get_SMOTE_folder_name, get_GAN_folder_name,  get_file_path, file_exists

def run_SMOTE(dataset_folder, dataset_name, target_column):
    new_file_name = get_new_file_name_SMOTE(dataset_name)
    new_file_path = get_file_path(dataset_folder, get_data_augmentation_folder_name(), get_SMOTE_folder_name(), new_file_name)
    if file_exists(new_file_path):
        print(f"File {new_file_name} already exists. Skipping conversion.")
        balanced_file_location = (dataset_folder, new_file_name)
    else:
        balanced_file_location = create_data_augmentation_SMOTE(dataset_folder, dataset_name, target_column)
    return balanced_file_location

def run_GAN(dataset_folder, dataset_name, target_column, minority_class, categorical_columns):
    new_file_name = get_new_file_name_GAN(dataset_name)
    new_file_path = get_file_path(dataset_folder, get_data_augmentation_folder_name(), get_GAN_folder_name(), new_file_name)
    if file_exists(new_file_path):
        print(f"File {new_file_name} already exists. Skipping conversion.")
        balanced_file_location = (dataset_folder, new_file_name)
    else:
        balanced_file_location = create_data_augmentation_GAN(dataset_folder, dataset_name, target_column, minority_class, categorical_columns)
    return balanced_file_location

### Use data augmnentation to create balanced datasets
def run_data_augmentation(imbalanced_file_locations, target_column, minority_class, categorical_columns):
    create_folder(imbalanced_file_locations[0][0], get_data_augmentation_folder_name())
    create_da_folder(imbalanced_file_locations[0][0], get_SMOTE_folder_name())
    create_da_folder(imbalanced_file_locations[0][0], get_GAN_folder_name())

    all_balanced_file_locations = []
    for file_location in imbalanced_file_locations:
        dataset_folder = file_location[0]
        dataset_name = file_location[1]

        # SMOTE
        SMOTE_file_location = run_SMOTE(dataset_folder, dataset_name, target_column)
        all_balanced_file_locations.append(SMOTE_file_location)

        # GAN
        GAN_file_location = run_GAN(dataset_folder, dataset_name, target_column, minority_class, categorical_columns)
        all_balanced_file_locations.append(GAN_file_location)

    return all_balanced_file_locations
