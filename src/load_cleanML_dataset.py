import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dataset(dataset_name, train_test_ratio_if_absent=0.7, random_seed_if_absent=42):
    """
    Loads train and test data for a given dataset. In case a train/test split is specified by the
    authors of the CleanML benchmark, train and test data is read as is. In case either train or
    test (dedicated) file is missing, the train and test data is read from `raw/raw.csv` and
    train/test split is performed using sklearn's implementation.

    Args:
        dataset_name (str): The name of the dataset.
        train_test_ratio_if_absent (float, optional): Train/test split ratio. Defaults to 0.7.
        Applies only if no train/test split is specified by the authors of the CleanML benchmark.
        random_seed_if_absent (int, optional): Random seed for train/test split. Defaults to 42.
        Applies only if no train/test split is specified by the authors of the CleanML benchmark.

    Returns:
        tuple: A tuple containing two pandas DataFrames (train_data, test_data).
    """
    dataset_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    train_file_path = os.path.join(dataset_path, 'dirty_train.csv')
    test_file_path = os.path.join(dataset_path, 'dirty_test.csv')
    raw_file_path = os.path.join(dataset_path, 'raw.csv')

    if os.path.exists(train_file_path) and os.path.exists(test_file_path):
        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)
    elif os.path.exists(raw_file_path):
        logger.info(f"No default train/test split found for {dataset_name}. "
                    f"Performing manual split with ratio {train_test_ratio_if_absent} and seed {random_seed_if_absent}.")
        data = pd.read_csv(raw_file_path)
        train_data, test_data = train_test_split(data, train_size=train_test_ratio_if_absent, random_state=random_seed_if_absent)
    else:
        raise FileNotFoundError(f"Neither train/test files nor raw file found for {dataset_name}.")

    return train_data, test_data
