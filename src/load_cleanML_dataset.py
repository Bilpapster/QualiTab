import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(filename)s::%(funcName)s::%(lineno)d %(asctime)s - %(levelname)s - %(message)s - ')
logger = logging.getLogger(__name__)


def prepare_data(train_data, test_data, target_column, random_seed):
    """Prepares the data by separating features and target, and limiting samples."""
    if target_column is None or target_column not in train_data.columns:
        target_column = train_data.columns[-1]
        logger.info(f"No or invalid target column specified. Using last column: {target_column}")

    MAX_SAMPLES = int(1e4)
    sampled_train = train_data.sample(n=min(MAX_SAMPLES, len(train_data)), replace=False, random_state=random_seed)

    X_train = sampled_train.drop(columns=[target_column])
    y_train = sampled_train[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    return X_train, y_train, X_test, y_test


def load_dataset(dataset_config: dict, mode: str = 'default', random_seed: int = 42):
    """
    Loads train and test data for a given dataset based on the provided configuration.
    Handles cases where train/test split files are missing by performing a manual split.

    Args:
        dataset_config (dict): A dictionary containing dataset-specific configurations with the following keys:
            - 'name' (str): The name of the dataset.
            - 'target_column' (str, optional): The name of the target column. If not provided or not found,
                                               the last column is used as the target.
            - 'train_test_ratio' (float, optional): The ratio for train/test split if manual split is needed.
                                                    Defaults to 0.7.

        mode (str): The mode for loading the dataset. Can be 'default' or 'force_manual_split'.
            `default` loads the dataset using the default train/test split if available, otherwise performs manual split.
            `force_manual_split` always performs a manual split regardless of the availability of default files.

        random_seed (int): The random seed for reproducibility in manual split. Defaults to 42.
                           If `default` mode is used and a train/test split is found, this value is ignored.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Features for training data.
            - y_train (pd.Series): Target for training data.
            - X_test (pd.DataFrame): Features for test data.
            - y_test (pd.Series): Target for test data.
            - used_default_split (bool): True if default train/test split was used, False if manual split was performed.
            - random_seed (int or None): The random seed used for manual split, or None if default split was used.
    """
    dataset_name = dataset_config['name']
    target_column = dataset_config.get('target_column')
    train_test_ratio = dataset_config.get('train_test_ratio', 0.7)
    random_seed = dataset_config.get('random_seed', 42)

    dataset_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    raw_file_path = os.path.join(dataset_path, 'raw.csv')
    if not os.path.isfile(raw_file_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_file_path}")

    if mode == 'default':
        train_file_path = os.path.join(dataset_path, 'dirty_train.csv')
        test_file_path = os.path.join(dataset_path, 'dirty_test.csv')

        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            train_data = pd.read_csv(train_file_path)
            test_data = pd.read_csv(test_file_path)
            used_default_split = True
            random_seed = None  # No random seed was used in this case
            X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, target_column, random_seed)
            return X_train, y_train, X_test, y_test, used_default_split, random_seed
        elif os.path.exists(raw_file_path):
            logger.info(f"No default train/test split found for {dataset_name}. "
                        f"Performing manual split with ratio {train_test_ratio} and seed {random_seed}.")

    # this part is reached if either mode is `force_manual_split` or default split was not found
    data = pd.read_csv(raw_file_path)
    train_data, test_data = train_test_split(data, train_size=train_test_ratio, random_state=random_seed)
    used_default_split = False
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, target_column, random_seed)
    return X_train, y_train, X_test, y_test, used_default_split, random_seed


from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
import time


def extract_embeddings(X_train, y_train, X_test, task):
    """Extracts embeddings using TabPFN."""
    start_time = time.time()

    if task == 'classification':
        model = TabPFNClassifier(n_estimators=1, random_state=42)
    elif task == 'regression':
        model = TabPFNRegressor(n_estimators=1, random_state=42)
    else:
        raise ValueError(f"Invalid task: {task}")

    # Construct the kwargs dictionary dynamically
    kwargs = {'n_fold': 0}  # Default n_fold
    if task == 'classification':
        kwargs['tabpfn_clf'] = model
    else:  # Regression
        kwargs['tabpfn_reg'] = model

    embedding_extractor = TabPFNEmbedding(**kwargs)
    train_embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_test, data_source="train")
    test_embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_test, data_source="test")
    end_time = time.time()
    return test_embeddings[0], start_time, end_time
