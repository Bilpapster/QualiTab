import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(filename)s::%(funcName)s::%(lineno)d %(asctime)s - %(levelname)s - %(message)s - ')
logger = logging.getLogger(__name__)

def load_raw_data(dataset_name):
    """Loads the raw.csv file for a given dataset."""
    raw_file_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw', 'raw.csv')
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Raw file not found for dataset {dataset_name}")
    return pd.read_csv(raw_file_path)

def perform_manual_split(data, train_test_ratio, random_seed):
    """Performs a manual train/test split."""
    return train_test_split(data, train_size=train_test_ratio, random_state=random_seed)

def load_default_split(dataset_name):
    """Loads the default train/test split if available."""
    train_file_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw', 'dirty_train.csv')
    test_file_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name, 'raw', 'dirty_test.csv')
    if os.path.exists(train_file_path) and os.path.exists(test_file_path):
        return pd.read_csv(train_file_path), pd.read_csv(test_file_path), f"{train_file_path} and {test_file_path}"
    return None, None, None

def prepare_data(train_data, test_data, target_column):
    """Prepares the data by separating features and target, and limiting samples."""
    if target_column is None or target_column not in train_data.columns:
        target_column = train_data.columns[-1]
        logger.info(f"No or invalid target column specified. Using last column: {target_column}")

    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    MAX_SAMPLES = int(1e4)
    X_train, y_train, X_test, y_test = X_train[:MAX_SAMPLES], y_train[:MAX_SAMPLES], X_test[:MAX_SAMPLES], y_test[:MAX_SAMPLES]
    return X_train, y_train, X_test, y_test


def load_dataset(dataset_config, mode='default'):
    """Loads the dataset based on the specified mode."""
    dataset_name = dataset_config['name']
    target_column = dataset_config.get('target_column')
    train_test_ratio = dataset_config.get('train_test_ratio', 0.7)
    random_seed = dataset_config.get('random_seed', 42)

    if mode == 'default':
        train_data, test_data, source_file = load_default_split(dataset_name)
        if train_data is None:  # Default split not found
            data = load_raw_data(dataset_name)
            train_data, test_data = perform_manual_split(data, train_test_ratio, random_seed)
            used_default_split = False
            source_file = "raw.csv (manual split)"  # Indicate manual split
        else:
            used_default_split = True
    elif mode == 'force_manual_split':
        data = load_raw_data(dataset_name)
        train_data, test_data = perform_manual_split(data, train_test_ratio, random_seed)
        used_default_split = False
        source_file = "raw.csv (manual split)"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, target_column)
    return X_train, y_train, X_test, y_test, used_default_split, random_seed, source_file


def check_and_load_clean_csv(file_path, raw_columns):
    """Checks if a CSV file matches the raw data columns and loads it."""
    try:
        clean_data = pd.read_csv(file_path)
        if set(clean_data.columns) == raw_columns:
            return clean_data
    except Exception as e:  # Handle potential errors during file loading
        logger.warning(f"Error loading or processing {file_path}: {e}")
    return None  # Return None if the file doesn't match or there's an error


def process_clean_file(file_path, raw_columns, dir_path, random_seed=42, train_size=0.7):
    """Processes a single clean data file, handling train/test splits."""
    clean_data = check_and_load_clean_csv(file_path, raw_columns)
    if clean_data is not None:
        if 'train' in file_path.lower():
            test_file = file_path.replace('train', 'test')
            if os.path.exists(test_file):
                test_data = check_and_load_clean_csv(test_file, raw_columns)
                if test_data is not None:
                    combined_data = pd.concat([clean_data, test_data], ignore_index=True)
                    train, test = perform_manual_split(combined_data, train_size, random_seed)
                    return train, test, f"{file_path} and {test_file}"
        else:  # No train/test split in filename
            train, test = perform_manual_split(clean_data, train_size, random_seed)
            return train, test, file_path
    return None, None, None  # Return None if processing fails


def scan_directory_for_clean_data(dir_path, raw_columns, random_seed=42, train_size=0.7):
    """Scans a directory for clean data files."""
    clean_data_sources = []
    for file in os.listdir(dir_path):
        if file.endswith('.csv') and 'clean' in file.lower():
            file_path = os.path.join(dir_path, file)
            train, test, source_info = process_clean_file(file_path, raw_columns, dir_path, random_seed, train_size)
            if train is not None:
                clean_data_sources.append((train, test, source_info, random_seed))
    return clean_data_sources


def scan_for_clean_data(dataset_name, random_seed=42, train_size=0.7):
    """Scans the dataset directory for clean data files."""
    base_path = os.path.join(os.pardir, 'CleanML', 'datasets', 'data', dataset_name)
    raw_data = load_raw_data(dataset_name)
    raw_columns = set(raw_data.columns)

    clean_data_sources = []

    # Scan raw directory
    raw_dir = os.path.join(base_path, 'raw')
    clean_data_sources.extend(scan_directory_for_clean_data(raw_dir, raw_columns, random_seed, train_size))

    # Scan other directories
    for dir_name in os.listdir(base_path):
        if dir_name != 'raw' and os.path.isdir(os.path.join(base_path, dir_name)):
            dir_path = os.path.join(base_path, dir_name)
            clean_data_sources.extend(scan_directory_for_clean_data(dir_path, raw_columns, random_seed, train_size))

    if not clean_data_sources:
        logger.warning(f"No clean data sources found for dataset {dataset_name}")

    return clean_data_sources