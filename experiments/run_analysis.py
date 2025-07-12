from src.evaluation import get_f1, shap_output

def unpack(model_info):
    model = model_info['model']
    training_data = model_info['training_data']
    testing_data = model_info['testing_data']
    return model, training_data, testing_data

def f1_score(model, testing_data):
    f1 = get_f1(model, testing_data)
    return f1

def all_analysis(models):
    for data_name in models:
        model, training_data, testing_data = unpack(models[data_name])
        f1 = f1_score(model, testing_data)
        # shap_output(model, training_data, testing_data)
        print(f"F1 Score for {data_name}: {f1:.4f}")