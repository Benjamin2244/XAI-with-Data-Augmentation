from .utils import read_csv_file
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


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
    
def train_model(X, y, num_features, num_classes, batch_size=32):
    model = NeuralNetwork(num_features, num_classes)
    model.train()  # Set the model to training mode

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_data = TensorDataset(X, y)
    generator = torch.Generator().manual_seed(24)
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=generator)
    # Training loop


    # Process one batch at a time, over multiple epochs
    num_epochs = 32
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
        if (epoch+1) % 10 == 0:
            average_loss = epoch_loss / len(training_loader)
            print(f'Epoch = {epoch+1} : Average Loss = {average_loss:.3f}')

    model.eval()  # Set the model to evaluation mode
    return model

def create_model(file_location, target_column):
    # Load dataset
    df = load_dataset(file_location)
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    # Parameter tuning
    # Train a model on the dataset
    num_features = X_train.shape[1]
    # num_classes = len(y_train.unique())
    num_classes = len(torch.unique(y_train))
    trained_model = train_model(X_train, y_train, num_features, num_classes)
    return trained_model, (X_test, y_test)