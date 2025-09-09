from sklearn.metrics import f1_score, confusion_matrix
import torch
import shap
from shap import Explanation
from src.utils import predict, get_parent_directory, get_results_folder_name
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

# Get the path for the SHAP results
def get_shap_results_path(file_location):
    dataset_folder, dataset_name = file_location
    parent_dir = get_parent_directory()
    path = parent_dir / 'data' / dataset_folder / get_results_folder_name() / f"{dataset_name}"
    return path

# Checks if the SHAP results exist
def does_shap_results_exist(file_location):
    dataset_folder, dataset_name = file_location
    parent_dir = get_parent_directory()
    path = parent_dir / 'data' / dataset_folder / get_results_folder_name() / f"{dataset_name}"
    if path.exists():
        return True
    return False

# Gets a background for SHAP to learn from
def get_background(X, y):
    test_size = 50 / len(y)
    background, _, _, _ = train_test_split(X, y, test_size=1 - test_size, stratify=y, random_state=24)
    return background

# Creates a DataFrame object
def create_DataFrame(data, feature_names):
    data = data.numpy()
    data = pd.DataFrame(data, columns=feature_names)
    return data

# Calculates the differences between two feature results
def calculate_feature_differences(result_A, result_B):
    A_np = np.array(result_A)
    B_np = np.array(result_B)
    differences = B_np - A_np
    return np.abs(differences)

# Calculates the mean SHAP values for each feature
def get_mean_shap_feature_values(shap_results):
    values = shap_results.values
    mean_values = np.mean(values, axis=0)
    return mean_values

# Calculates the euclidean distance between scores.
def calculate_euclidean(differences):
    difference = np.linalg.norm(differences)
    return difference

# Compares results to the control results
def compare_results(control, data_augmentated_results):
    all_feature_differences = []
    all_total_differences = []

    for data_augmentated_result in data_augmentated_results:
        control_feature_values_means = get_mean_shap_feature_values(control)
        data_augmentated_feature_values_means = get_mean_shap_feature_values(data_augmentated_result)

        feature_differences = calculate_feature_differences(control_feature_values_means, data_augmentated_feature_values_means)
        all_feature_differences.append(feature_differences)
        total_differences = calculate_euclidean(feature_differences)
        all_total_differences.append(total_differences)

    return all_feature_differences, all_total_differences

# Gets the colours for the results
def get_colours(values):
    values_copy = np.abs(values.copy())

    first_most_different_index = 0
    second_most_different_index = 0
    third_most_different_index = 0

    first_most_different_index = np.argmax(values_copy)
    values_copy[first_most_different_index] = -np.inf
    if len(values) >= 2:
        second_most_different_index = np.argmax(values_copy)
        values_copy[second_most_different_index] = -np.inf
        if len(values) >= 3:
            third_most_different_index = np.argmax(values_copy)

    light_grey = '#A1A1A1' # Alternating
    dark_grey = '#525252' # Alternating
    highlight = "#406C19" # Top three results

    colours = [light_grey if i % 2 == 0 else dark_grey for i in range(len(values))]

    for index in [first_most_different_index, second_most_different_index, third_most_different_index]:
        colours[index] = highlight

    return colours

# Display the feature scores in comparison to the control
def display_feature_differences(differences, order, dataset, features):
    for feature_differences, model in zip(differences, order): 
        colours = get_colours(feature_differences)

        plt.bar(features, feature_differences, color=colours)

        plt.xlabel('Features')
        plt.xticks(rotation='vertical')

        plt.ylabel('Difference from control')

        plt.title(f"Dataset: {dataset} with {model}")
        plt.show()

# Display the overall model difference in comparison to the control
def display_total_differences(differences, order, dataset, DA_method):
    colours = get_colours(differences)

    plt.bar(order, differences, color=colours)

    plt.xlabel('Models')
    plt.xticks(rotation='vertical')

    plt.ylabel('Difference from control')

    plt.title(f"Dataset: {dataset} with {DA_method}")
    plt.show()

# Show the SHAP result for individual rows
def show_shap_result(shap_results, feature_names, index):
    shap_result = shap_results[index]

    shap_result_for_row = shap.Explanation(
        values=shap_result,
        base_values=shap_results.base_values[index],
        data=shap_results.data[index],
        feature_names=feature_names
    )

    shap.plots.waterfall(shap_result_for_row)


def split_shap_results(shap_results):
    values = shap_results.values
    base_values = shap_results.base_values
    data = shap_results.data
    feature_names = shap_results.feature_names
    predictions = base_values + values.sum(axis=1)

    minority_indicies = predictions >= 0.5
    majority_indicies = predictions < 0.5

    minority_explanations = Explanation(
        values = values[minority_indicies],
        base_values = base_values[minority_indicies],
        data = data[minority_indicies],
        feature_names = feature_names,
    )

    majority_explanations = Explanation(
        values = values[majority_indicies],
        base_values = base_values[majority_indicies],
        data = data[majority_indicies],
        feature_names = feature_names,
    )
    return  minority_explanations, majority_explanations

# Use SHAP's built in displays for results
def show_shap_results(shap_values, feature_names):
    minority_explanations, majority_explanations = split_shap_results(shap_values)

    # Changes the scale from 0-1 to 0-100
    shap_values.values = shap_values.values * 100
    minority_explanations.values = minority_explanations.values * 100
    majority_explanations.values = majority_explanations.values * 100

    max_display = len(feature_names)
    if max_display > 20:
        max_display = 15 # Used for forest cover

    # SHAP results for the overall model
    shap.plots.bar(shap_values, max_display=max_display)
    shap.plots.beeswarm(shap_values, max_display=max_display)

    # SHAP results for the minority class classification
    shap.plots.bar(minority_explanations, max_display=max_display)
    shap.plots.beeswarm(minority_explanations, max_display=max_display)

    # SHAP results for the majority class classification
    shap.plots.bar(majority_explanations, max_display=max_display)
    shap.plots.beeswarm(majority_explanations, max_display=max_display)

    # # SHAP results for all individual samples
    # for sample in range(len(shap_values)):
    #     shap.plots.waterfall(shap_values[sample], max_display=len(feature_names))

    # SHAP results for spesific samples
    shap.plots.waterfall(shap_values[0], max_display=max_display)
    shap.plots.waterfall(shap_values[1], max_display=max_display)
    shap.plots.waterfall(shap_values[73], max_display=max_display) # This is for a minority sample for forest cover

# Create the SHAP explainer
def create_explainer(data, model):
    model.eval()
    masker = shap.maskers.Independent(data)
    model_prediction = functools.partial(predict, model)
    explainer = shap.Explainer(model_prediction, masker)
    return explainer

# Calculate the SHAP scores
def get_shap_scores(shap_results, class_options):
    shap_values = shap_results.values

    shap_values_dict = {}
    importance_avg = np.abs(shap_values).mean(axis=0)
    for index, class_value in enumerate(class_options):
        values = []
        for avg in importance_avg:
            values.append(avg[index])
        shap_values_dict[class_value] = values
    
    return shap_values_dict

# Completes SHAP from start to finish
def shap_explainer(model, testing_data, target_column, minority_class, feature_names, dataset_folder, data_name):
    X_test, y_test = testing_data
    X_test_df = pd.DataFrame(X_test.numpy(), columns=feature_names)

    results_file = f"{data_name.removesuffix('.csv')}.pk1"
    shap_results_path = get_shap_results_path((dataset_folder, f"{data_name.removesuffix('.csv')}.pk1"))

    if does_shap_results_exist((dataset_folder, results_file)):
        print(f"Results for {data_name} already exists. Loading the results.")
        shap_values = joblib.load(shap_results_path)
    else:
        print(f"Results for {data_name} does not exist. Calculating new results.")
        prediction_fn = lambda X: predict(model, X)
        explainer = shap.Explainer(prediction_fn, X_test_df)
        shap_values = explainer(X_test_df)
        joblib.dump(shap_values, shap_results_path)

    return shap_values    

# Get the f1 score
def get_f1(model, testing_data):
    X_test, y_test = testing_data
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, dim=1)

    y_true = y_test.cpu().numpy()
    y_prediction = predictions.cpu().numpy()

    f1 = f1_score(y_true, y_prediction, average='macro')
    return f1

# Show a confusion matrix
def show_confusion_matrix(model, testing_data):
    X_test, y_test = testing_data
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, dim=1)

    y_true = y_test.cpu().numpy()
    y_prediction = predictions.cpu().numpy()

    results = confusion_matrix(y_true, y_prediction, labels=[0,1])
    TN, FP, FN, TP = results.ravel()

    confusion_matrix_display = pd.DataFrame([[TN, FP], [FN, TP]], columns=["Prediction 0", "Prediction 1"], index=["True 0", "True 1"])
    print(confusion_matrix_display)
