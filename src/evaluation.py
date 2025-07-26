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

def get_background(training_data):
    X_train, y_train = training_data
    background = X_train[:50]
    return background

def shap_output(model, training_data, test_data):
    background = get_background(training_data)
    X_test, y_test = test_data
    interpreter = shap.GradientExplainer(model, background)
    shap_values = interpreter.shap_values(X_test)
    shap.summary_plot(shap_values, X_test.cpu().numpy())

def create_DataFrame(data, feature_names):
    data = data.numpy()
    data = pd.DataFrame(data, columns=feature_names)
    return data

def create_explainer(data, model):
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

def display_feature_differences(differences, order, dataset, features):
    for (class_value, all_model_values) in differences:
        print(class_value)
        for values, model in zip(all_model_values, order):
            plt.bar(features, values)
            plt.xlabel('Features')
            plt.ylabel('Difference from control')
            plt.title(f"Dataset: {dataset} with {model}, looking at class {class_value}")
            plt.show()


def display_total_differences(differences, order, dataset, features):
    pass

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
    X_test = create_DataFrame(X_test, feature_names)
    class_options = sorted(set(y_test.numpy()))
    explainer = create_explainer(X_test, model)

    shap_results = explainer(X_test)
    shap_results.feature_names = feature_names

    shap_values_dict = get_shap_scores(shap_results, class_options)

    return shap_values_dict


    # TODO save the dict with in an new dict, with a new key, prob the model name
    # TODO return the new dict so i can save it and then compare with others
    # TODO have a utils func for euclidean distance?
    # TODO compare the control with each other indiviuallly, using euclidean
    # TODO print results in a seperate funct
    # TODO have a graph showing each model and there difference from the original
    # TODO compare all models together? have in that weird box thing? sees smote vs gan

    

    # show_shap_results(shap_results, feature_names, y_test, minority_class) # Uncomment for shap default graph