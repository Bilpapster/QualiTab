import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_name, train_test_ratio=0.7, random_seed=42, target_column=None):
    """
    Loads train and test data for a given dataset. In case a train/test split is specified by the
    authors of the CleanML benchmark, train and test data is read as is. In case either train or
    test (dedicated) file is missing, the train and test data is read from `raw/raw.csv` and
    train/test split is performed using sklearn's implementation.

    Args:
        dataset_name (str): The name of the dataset.
        train_test_ratio (float, optional): Train/test split ratio (if manual split). Defaults to 0.7.
        random_seed (int, optional): Random seed for train/test split (if manual split). Defaults to 42.
        target_column (str, optional): The name of the target column. If None or the specified column name
        does not exist in the column names, the last column is used.

    Returns:
        tuple: A tuple containing four pandas DataFrames (X_train, y_train, X_test, y_test).
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
                    f"Performing manual split with ratio {train_test_ratio} and seed {random_seed}.")
        data = pd.read_csv(raw_file_path)
        train_data, test_data = train_test_split(data, train_size=train_test_ratio, random_state=random_seed)
    else:
        raise FileNotFoundError(f"Neither train/test files nor raw file found for {dataset_name}.")

    # Handle target column
    if target_column is None or target_column not in train_data.columns:
        if target_column is not None:
            logger.warning(f"Specified target column '{target_column}' not found for dataset '{dataset_name}'. "
                           f"Using the last column as target. Available columns {train_data.columns.values}")
        else:
            logger.info(f"No target column specified for dataset '{dataset_name}'. Using the last column as target.")
        target_column = train_data.columns[-1]  # Use the last column

    logger.info(f"Target column set to: '{target_column}'")

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    return X_train, y_train, X_test, y_test
