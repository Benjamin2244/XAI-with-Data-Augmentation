from .utils import read_csv_file, print_progress_dot_optuna
from .evaluation import get_f1
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import optuna
import functools



def load_dataset(file_location):
    dataset_folder, dataset_file_name = file_location
    df = read_csv_file(dataset_folder, dataset_file_name)
    return df

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=24)

    X_train = X_train.astype(float)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)

    X_test = X_test.astype(float)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    return X_train, X_test, y_train, y_test

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

# Hyperparameter tuning with Optuna
def hyperparameter_optimisation(X_train, y_train, X_test, y_test, num_features, num_classes):
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
    study = optuna.create_study(direction='maximize')
    study.optimize(partial_objective, n_trials=100, callbacks=[print_progress_dot_optuna])

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

def create_model(file_location, target_column):
    df = load_dataset(file_location) # Load dataset
    X_train, X_test, y_train, y_test = split_data(df, target_column) # Split data
    num_features = X_train.shape[1] # Train a model on the dataset
    num_classes = len(torch.unique(y_train))

    parameters = hyperparameter_optimisation(X_train, y_train, X_test, y_test, num_features, num_classes) # Parameter tuning

    trained_model = train_model(
        X_train, y_train, 
        num_features, num_classes,
        lr=parameters['lr'],
        batch_size=parameters['batch_size'],
        num_epochs=32)
    
    return trained_model, (X_train, y_train), (X_test, y_test)