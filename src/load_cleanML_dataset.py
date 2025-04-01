import os
import pandas as pd


def load_dataset(dataset_name):
    """
    Load the train and test data for a given dataset from the CleanML directory.

    Parameters:
    dataset_name (str): The name of the dataset (e.g., 'Airbnb').

    Returns:
    tuple: A tuple containing two pandas DataFrames (train_data, test_data).
    """
    # Construct the path to the dataset directory
    dataset_path = os.path.join(os.path.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw')

    # Check if the directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset directory {dataset_path} does not exist.")

    # Construct the paths to the train and test files
    train_file_path = os.path.join(dataset_path, 'dirty_train.csv')
    test_file_path = os.path.join(dataset_path, 'dirty_test.csv')

    # Check if the train file exists
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"The train file {train_file_path} does not exist.")

    # Check if the test file exists
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"The test file {test_file_path} does not exist.")

    # Read the CSV files into pandas DataFrames
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    return train_data, test_data


# Example usage
if __name__ == "__main__":
    dataset_name = 'Airbnb'
    train_df, test_df = load_dataset(dataset_name)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")