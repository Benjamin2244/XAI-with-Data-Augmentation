from pathlib import Path
import pandas as pd


# Get parent directory of the current file
def get_parent_directory():
    parent_dir = Path(__file__).parent
    return parent_dir.parent

# Read the CSV file
def read_csv_file(dataset_folder, file):
    parent_dir = get_parent_directory()
    csv_file = parent_dir / 'data' / dataset_folder / file
    if not csv_file.exists():
        raise FileNotFoundError(f"File {csv_file} does not exist.")
    return pd.read_csv(csv_file)