from src.evaluation import get_f1

def all_analysis(models):
    for data_name in models:
        model = models[data_name]['model']
        testing_data = models[data_name]['testing_data']
        f1 = get_f1(model, testing_data)
        print(f"F1 Score for {data_name}: {f1:.4f}")