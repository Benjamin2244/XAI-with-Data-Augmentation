from sklearn.metrics import f1_score
import torch
import shap
import torch.nn as nn
from src.utils import predict
import functools




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

def shap_explainer(model, testing_data):
    X_test, y_test = testing_data
    X_test = X_test.numpy()

    # TODO FIX this
    # Create fake feature names
    num_features = X_test.shape[1]
    feature_names = [f'feat_{i}' for i in range(num_features)]

    import pandas as pd
    X_test = pd.DataFrame(X_test, columns=feature_names)
    # TODO FIX ABOVE

    masker = shap.maskers.Independent(X_test)

    model_prediction = functools.partial(predict, model)
    explainer = shap.Explainer(model_prediction, masker)

    shap_results = explainer(X_test)

    shap_results.feature_names = feature_names
    # change below
    print("shap_results.values.shape:", shap_results.values.shape)
    print("Number of feature names:", len(feature_names))
    # Select the class index to explain (e.g., class 1)
    class_idx = 1  
    shap_values_for_class = shap_results.values[:, :, class_idx]

    # Create a new Explanation object containing only SHAP values for that class
    shap_result_for_class = shap.Explanation(
        shap_values_for_class,
        base_values=shap_results.base_values[:, class_idx],
        data=shap_results.data,
        feature_names=feature_names
    )
    # Change above

    shap.plots.bar(shap_result_for_class)