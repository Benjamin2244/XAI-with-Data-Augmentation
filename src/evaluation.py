from sklearn.metrics import f1_score
import torch
import shap
import torch.nn as nn



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