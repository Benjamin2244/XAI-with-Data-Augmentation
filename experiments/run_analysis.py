from src.evaluation import get_f1, shap_output
from src.utils import load_test_data

def f1_score(model, testing_data):
    f1 = get_f1(model, testing_data)
    return f1

def all_analysis(models):
    for data_name in models:
        model_dict = models[data_name]
        model = model_dict['model']
        dataset = model_dict['dataset']
        target_column = model_dict['target_column']

        testing_data = load_test_data(dataset, target_column)

        f1 = f1_score(model, testing_data)
        # shap_output(model, training_data, testing_data)
        print(f"F1 Score for {data_name}: {f1:.4f}")