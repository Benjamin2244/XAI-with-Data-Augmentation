from sklearn.metrics import f1_score
import torch
import shap
import torch.nn as nn
from src.utils import predict
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def get_f1(model, testing_data):
    X_test, y_test = testing_data
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, dim=1)

    y_true = y_test.cpu().numpy()
    y_prediction = predictions.cpu().numpy()

    f1 = f1_score(y_true, y_prediction, average='macro')
    return f1

def get_background(X):
    shuffled_X = X[torch.randperm(X.size(0))]
    background = shuffled_X[:50]
    return background

# def get_background(training_data):
#     X_train, y_train = training_data
#     print(type(X_train))
#     background = X_train[:50]
#     return background

# def shap_output(model, training_data, test_data):
#     background = get_background(training_data)
#     X_test, y_test = test_data
#     interpreter = shap.GradientExplainer(model, background)
#     shap_values = interpreter.shap_values(X_test)
#     shap.summary_plot(shap_values, X_test.cpu().numpy())

def create_DataFrame(data, feature_names):
    data = data.numpy()
    data = pd.DataFrame(data, columns=feature_names)
    return data

def create_explainer(data, model):
    model.eval()
    masker = shap.maskers.Independent(data)
    model_prediction = functools.partial(predict, model)
    explainer = shap.Explainer(model_prediction, masker)
    return explainer

def show_shap_results(shap_results, feature_names, y_test, class_value):
    class_options = sorted(set(y_test.numpy()))
    class_index = class_options.index(class_value)

    shap_values_for_class = shap_results.values[:, :, class_index]

    shap_result_for_class = shap.Explanation(
        shap_values_for_class,
        base_values=shap_results.base_values[:, class_index],
        data=shap_results.data,
        feature_names=feature_names
    )

    shap.plots.bar(shap_result_for_class, max_display=len(feature_names))
    shap.plots.beeswarm(shap_result_for_class, max_display=len(feature_names))

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

def calculate_feature_differences(result_A, result_B):
    A_np = np.array(result_A)
    B_np = np.array(result_B)
    differences = B_np - A_np
    return differences

def calculate_euclidean(differences):
    difference = np.linalg.norm(differences)
    return difference

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

    light_grey = '#A1A1A1'
    dark_grey = '#525252'
    highlight = "#406C19"

    colours = [light_grey if i % 2 == 0 else dark_grey for i in range(len(values))]

    for index in [first_most_different_index, second_most_different_index, third_most_different_index]:
        colours[index] = highlight

    return colours

def display_feature_differences(differences, order, dataset, features):
    for (class_value, all_model_values) in differences:
        for values, model in zip(all_model_values, order):
            colours = get_colours(values)

            plt.bar(features, values, color=colours)

            plt.xlabel('Features')
            plt.xticks(rotation='vertical')

            plt.ylabel('Difference from control')

            plt.title(f"Dataset: {dataset} with {model}, looking at class {class_value}")
            plt.show()


def display_total_differences(differences, order, dataset, DA_method):
    print(F"differences: {differences}")
    for (class_value, model_differences) in differences:
        colours = get_colours(model_differences)

        plt.bar(order, model_differences, color=colours)

        plt.xlabel('Models')
        plt.xticks(rotation='vertical')

        plt.ylabel('Difference from control')

        plt.title(f"Dataset: {dataset} with {DA_method}, looking at class {class_value}")
        plt.show()

def compare_results(control, results):
    feature_differences = []
    total_differences = []

    for class_option in control:
        control_result = control[class_option]
        class_feature_differences = []
        class_total_differences = []

        for result in results:
            differences = calculate_feature_differences(control_result, result[class_option])
            class_feature_differences.append(differences)

            difference = calculate_euclidean(differences)
            class_total_differences.append(difference)
        
        feature_differences.append((class_option, class_feature_differences))
        total_differences.append((class_option, class_total_differences))

    return feature_differences, total_differences
            

def shap_explainer(model, testing_data, target_column, minority_class, feature_names):
    X_test, y_test = testing_data
    X_subset = get_background(X_test)
    X_subset = create_DataFrame(X_subset, feature_names)
    class_options = sorted(set(y_test.numpy()))

    explainer = create_explainer(X_subset, model)

    X_test = create_DataFrame(X_test, feature_names)
    shap_results = explainer(X_test)
    shap_results.feature_names = feature_names

    shap_values_dict = get_shap_scores(shap_results, class_options)

    return shap_values_dict


    # TODO compare all models together? have in that weird box thing? sees smote vs gan
    # TODO smaller background

    

    # show_shap_results(shap_results, feature_names, y_test, minority_class) # Uncomment for shap default graph