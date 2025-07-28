from src.evaluation import get_f1, shap_explainer, compare_results, display_feature_differences, display_total_differences
from src.utils import load_test_data, load_column_names, is_control, is_SMOTE, is_GAN

def f1_score(model, testing_data):
    f1 = get_f1(model, testing_data)
    return f1

def do_shap(model, testing_data, target_column, minority_class, feature_names):
    values = shap_explainer(model, testing_data, target_column, minority_class, feature_names)
    return values

def show_SHAP_differences(feature_differences, total_difference, order, dataset, feature_names, DA_method):
    display_feature_differences(feature_differences, order, dataset, feature_names)
    display_total_differences(total_difference, order, dataset, DA_method)

def analyse_shap(all_shap_results, all_feature_names):
    for dataset in all_shap_results:
        dataset_results = all_shap_results[dataset]
        feature_names = all_feature_names[dataset]
        control_results = None
        SMOTE_results = []
        SMOTE_order = []
        GAN_results = []
        GAN_order = []
        for model in dataset_results:
            if is_control(model):
                control_results = dataset_results[model]
            elif is_SMOTE(model):
                SMOTE_results.append(dataset_results[model])
                SMOTE_order.append(model)
            elif is_GAN(model):
                GAN_results.append(dataset_results[model])
                GAN_order.append(model)
        if control_results == None:
            continue
        if (len(SMOTE_results) == 0) or (len(GAN_results) == 0):
            continue

        SMOTE_feature_differences, SMOTE_total_difference = compare_results(control_results, SMOTE_results)
        GAN_feature_differences, GAN_total_difference = compare_results(control_results, GAN_results)

        show_SHAP_differences(SMOTE_feature_differences, SMOTE_total_difference, SMOTE_order, dataset, feature_names, "SMOTE")
        show_SHAP_differences(GAN_feature_differences, GAN_total_difference, GAN_order, dataset, feature_names, "GAN")

def all_analysis(models):
    all_shap_results = {}
    all_feature_names = {}
    for data_name in models:
        model_dict = models[data_name]
        model = model_dict['model']
        dataset = model_dict['dataset']
        target_column = model_dict['target_column']
        minority_class = model_dict['minority_class']

        testing_data = load_test_data(dataset, target_column)
        feature_names = load_column_names(dataset, target_column)

        if dataset not in all_shap_results:
            all_shap_results[dataset] = {}
            all_feature_names[dataset] = feature_names

        f1 = f1_score(model, testing_data)
        print(f"F1 Score for {data_name}: {f1:.4f}")

        values = do_shap(model, testing_data, target_column, minority_class, feature_names)
        all_shap_results[dataset][data_name] = values

    analyse_shap(all_shap_results, all_feature_names)
        