from .utils import read_csv_file, print_progress_dot_optuna, get_model_folder_name, get_parent_directory, get_pre_data_augmentation_folder_name, force_csv_extension, force_pt_extension, load_model, split_data, load_dataset, get_num_features, get_num_classes, create_param_folder, print_progress_dot
from .evaluation import get_f1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
import functools

class NeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
# Hyperparameter tuning with Optuna
def objective(trial, X_train, y_train, X_test, y_test, num_features, num_classes):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    model = train_model(
        X_train, y_train,
        num_features, num_classes,
        lr=lr, batch_size=batch_size,
        num_epochs=32
    )

    f1 = get_f1(model, (X_test, y_test))
    return f1

def does_trial_count_exist(path):
    if path.exists():
        return True
    return False

def create_trial_count(path):
    with open(path, "w") as file:
        file.write("0")

def get_trial_count(path):
    with open(path, "r") as file:
        return int(file.read().strip())

def update_trial_count(path, count):
    with open(path, "w") as file:
        file.write(str(count))

# Hyperparameter tuning with Optuna
def hyperparameter_optimisation(X_train, y_train, X_test, y_test, num_features, num_classes, save_progress, file_location):
    dataset_folder, dataset_name = file_location

    partial_objective = functools.partial(
        objective, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        num_features=num_features, 
        num_classes=num_classes
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Stops default logging

    if save_progress:
        param_details_path = create_param_folder(dataset_folder)

        trial_count_path = param_details_path / f"trial_count_for_{dataset_name}.txt"
        if does_trial_count_exist(trial_count_path) == False:
            create_trial_count(trial_count_path)
        trial_count = get_trial_count(trial_count_path)

        optuna_path = param_details_path / f"params_for_{dataset_name}.db"
        optuna_path_URI = f"sqlite:///{optuna_path}"

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=24), study_name=f"{dataset_name}_study", 
                                        storage=optuna_path_URI, load_if_exists=True)
        
        for _ in range(trial_count):
            print_progress_dot()

        while trial_count < 25:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=24), study_name=f"{dataset_name}_study", 
                                        storage=optuna_path_URI, load_if_exists=True)

            study.optimize(partial_objective, n_trials=1, callbacks=[print_progress_dot_optuna])

            trial_count += 1
            update_trial_count(trial_count_path, trial_count)

    else:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=24))
        study.optimize(partial_objective, n_trials=100, callbacks=[print_progress_dot_optuna])
    print()

    return study.best_params

def train_model(X, y, num_features, num_classes, lr, batch_size, num_epochs):
    model = NeuralNetwork(num_features, num_classes)
    model.train()  # Set the model to training mode

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_data = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(24)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=generator)
    
    # Training loop
    # Process one batch at a time, over multiple epochs
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in training_loader:  
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y) # Calculate loss
            loss.backward() # Update model
            epoch_loss += loss.item()
            optimizer.step()
        
        # Print average loss every 10 epochs
        # if (epoch+1) % 10 == 0:
        #     average_loss = epoch_loss / len(training_loader)
        #     print(f'Epoch = {epoch+1} : Average Loss = {average_loss:.3f}')

    model.eval()  # Set the model to evaluation mode
    return model

def does_model_exist(file_location):
    dataset_folder, dataset_name = file_location
    parent_dir = get_parent_directory()
    path = parent_dir / 'data' / dataset_folder / get_model_folder_name() / f"{dataset_name}"
    if path.exists():
        return True
    return False

def save_model(model, file_location):
    dataset_folder, dataset_name = file_location
    parent_dir = get_parent_directory()
    path = parent_dir / 'data' / dataset_folder / get_model_folder_name() / f"{dataset_name.removesuffix('.csv')}.pt"
    torch.save(model.state_dict(), path)

def create_model(file_location, target_column, dataset_type, da_subfolder):
    df = load_dataset(file_location, dataset_type, da_subfolder) # Load dataset
    X_train, X_val, y_train, y_val = split_data(df, target_column) # Split data
    num_features = get_num_features(file_location, target_column, dataset_type, da_subfolder)
    num_classes = get_num_classes(file_location, target_column, dataset_type, da_subfolder)
    save_progress = False
    if len(df) >= 100000:
        save_progress = True

    parameters = hyperparameter_optimisation(X_train, y_train, X_val, y_val, num_features, num_classes, save_progress, file_location) # Parameter tuning

    trained_model = train_model(
        X_train, y_train, 
        num_features, num_classes,
        lr=parameters['lr'],
        batch_size=parameters['batch_size'],
        num_epochs=32)
    
    save_model(trained_model, file_location)

    return trained_model