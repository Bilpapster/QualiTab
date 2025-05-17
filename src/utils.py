import logging
from dotenv import load_dotenv
import openml
import pandas as pd

from config import (
    TABPFN_MAX_SAMPLES,
    TABPFN_MAX_FEATURES,
    TABPFN_MAX_CLASSES,
    openML_dataset_configs
)


def configure_logging() -> logging.Logger:
    """
    Configures the logging for the application.

    This function sets up the basic configuration for logging,
    specifying the log level, format, and other settings.
    It then returns a logger instance that can be used throughout the application.

    Returns:
        logging.Logger: A logger instance configured with the specified settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s - '
    )
    return logging.getLogger(__name__)


def parse_comma_separated_integers(s):
    """
    Converts a comma-separated string of integers into a list of ints.
    Example: "100,200,300..." -> [100, 200, 300, ...]
    """
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.strip().split(',')]


def get_seeds_from_env_or_else_default() -> list[int]:
    """
    Retrieves a list of random seeds from the environment variable 'SEEDS'.
    If the environment variable is not set, it returns a default list of seeds.
    """
    import os

    load_dotenv()
    seeds_str = os.getenv('SEEDS', '100,200,300')
    return parse_comma_separated_integers(seeds_str)


def get_datasets_to_skip_from_env_or_else_empty() -> set[int]:
    """
    Retrieves a list of datasets to skip from the environment variable 'DATASETS_TO_SKIP'.
    If the environment variable is not set, it returns an empty set.
    """
    import os

    load_dotenv()
    datasets_to_skip_str = os.getenv('DATASETS_TO_SKIP', "")
    return set(parse_comma_separated_integers(datasets_to_skip_str))


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def connect_to_db() -> tuple:
    """
    Connects to the PostgreSQL database and returns a tuple
    (connection, connection_cursor).

    example: conn, cursor = connect_to_db()
    """
    import psycopg2
    import os

    load_dotenv()
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            port=os.getenv("POSTGRES_MAPPED_PORT", "5432"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            database=os.getenv("POSTGRES_DB", "postgres"),
        )
        return conn, conn.cursor()
    except psycopg2.OperationalError:
        raise "Error connecting to the database. Please check your connection settings."


def get_GPU_information() -> list[tuple]:
    import torch
    GPUs = []
    for i in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(i)
        GPUs.append((properties.name, properties.total_memory))
    return GPUs


def get_openML_task_mapping(benchmark_name="OpenML-CC18") -> dict:
    """
    Retrieves the mapping of OpenML dataset IDs to task IDs for a given benchmark name.
    Args:
        benchmark_name (str): The name of the benchmark suite to retrieve tasks from.
    Returns:
        dict: A dictionary mapping dataset IDs to task IDs. Note that database stores dataset IDs but OpenML uses
        task IDs to retrieve the data.
    """
    import openml

    tasks = openml.study.get_suite(benchmark_name).tasks
    mapping = dict()

    for task_id in tasks:
        task = openml.tasks.get_task(task_id)
        dataset_id = task.dataset_id
        mapping[dataset_id] = task_id

    return mapping


def print_openML_report_wrt_tabpfn_limits():
    benchmark_configs = openML_dataset_configs
    for benchmark_config in benchmark_configs:
        benchmark_suite = openml.study.get_suite(benchmark_config['name'])
        nof_tasks = len(benchmark_suite.tasks)
        nof_datasets_with_too_many_samples = 0
        nof_datasets_with_too_many_train_samples = 0 # assuming train size is 70% of the dataset
        nof_datasets_with_too_many_test_samples = 0 # assuming test size is 30% of the dataset
        nof_datasets_with_too_many_features = 0
        nof_datasets_with_too_many_targets = 0

        print(f"--------- BENCHMARK SUITE {benchmark_config['description']} ---------")
        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)
            features, targets = task.get_X_and_y(dataset_format='dataframe')
            if len(features) > TABPFN_MAX_SAMPLES:
                nof_datasets_with_too_many_samples += 1
            if len(features) * 0.7 > TABPFN_MAX_SAMPLES:
                nof_datasets_with_too_many_train_samples += 1
            if len(features) * 0.3 > TABPFN_MAX_SAMPLES:
                nof_datasets_with_too_many_test_samples += 1
            if len(features.columns) > TABPFN_MAX_FEATURES:
                nof_datasets_with_too_many_features += 1
            if benchmark_config['task'] == 'classification' and len(set(targets)) > TABPFN_MAX_CLASSES:
                nof_datasets_with_too_many_targets += 1

        print(f"\tTasks: {nof_tasks:20}")
        print(f"\tToo many samples: {nof_datasets_with_too_many_samples:9} ({nof_datasets_with_too_many_samples / nof_tasks:.2%})")
        print(f"\tToo many train samples: {nof_datasets_with_too_many_train_samples:3} ({nof_datasets_with_too_many_train_samples / nof_tasks:.2%})")
        print(f"\tToo many test samples: {nof_datasets_with_too_many_test_samples:4} ({nof_datasets_with_too_many_test_samples / nof_tasks:.2%})")
        print(f"\tToo many features: {nof_datasets_with_too_many_features:8} ({nof_datasets_with_too_many_features / nof_tasks:.2%})")
        if benchmark_config['task'] == 'classification':
            print(f"\tToo many targets: {nof_datasets_with_too_many_targets:9} ({nof_datasets_with_too_many_targets / nof_tasks:.2%})")
        print()

def fetch_baseline_metric_value(
    conn, evaluation_type: str = None, metric_name: str = None, tag: str = 'CLEAN_CLEAN',
    train_size_max: int = None, train_size_min: int = None,
    test_size_max: int = None, test_size_min: int = None
):
    cursor = conn.cursor()
    query = f"""
    SELECT AVG(metric_value) AS baseline_metric_value
    FROM embeddings_experiments ex JOIN embedding_evaluation_metrics ev
    ON ex.experiment_id = ev.experiment_id
    WHERE ex.tag = '{tag}'
    """
    query += append_parameters_to_query(
        evaluation_type=evaluation_type,
        metric_name=metric_name,
        train_size_max=train_size_max,
        train_size_min=train_size_min,
        test_size_max=test_size_max,
        test_size_min=test_size_min
    )

    cursor.execute(query)
    return cursor.fetchall()[0][0]


def fetch_corrupted_metric_values(
        conn, evaluation_type: str = None, metric_name: str = None, tag: str = 'DIRTY_DIRTY',
        train_size_max: int = None, train_size_min: int = None,
        test_size_max: int = None, test_size_min: int = None
):
    import pandas as pd

    cursor = conn.cursor()
    query = f"""
    SELECT CONCAT(ex.error_type, '_', ex.row_corruption_percent) as type_rate, AVG(metric_value) AS avg_metric_value_error
    FROM embeddings_experiments ex JOIN embedding_evaluation_metrics ev
    ON ex.experiment_id = ev.experiment_id
    WHERE ex.tag = '{tag}'
    """

    query += append_parameters_to_query(
        evaluation_type=evaluation_type,
        metric_name=metric_name,
        train_size_max=train_size_max,
        train_size_min=train_size_min,
        test_size_max=test_size_max,
        test_size_min=test_size_min
    )

    query += " GROUP BY type_rate"

    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=columns)
    cursor.close()
    # Split the type_rate into error_type and corruption_rate
    df['error_type'] = df['type_rate'].str.split('_').str[0]
    df['corruption_rate'] = df['type_rate'].str.split('_').str[1].astype(int)

    # Convert avg_metric_value_error to float
    df['avg_metric_value_error'] = df['avg_metric_value_error'].astype(float)

    # Drop the original type_rate column and rename avg_metric_value_error to metric_value
    df = df.drop(columns=['type_rate'])
    df = df.rename(columns={'avg_metric_value_error': 'metric_value'})

    # Sort the DataFrame by error_type and (secondly) corruption_rate within each error_type
    result = {}
    for error_type, group in df.groupby('error_type'):
        # Sort again to ensure order is maintained
        group = group.sort_values('corruption_rate')
        result[error_type] = {
            'rates': group['corruption_rate'].tolist(),
            'values': group['metric_value'].tolist()
        }
    return result

def append_parameters_to_query(
    evaluation_type: str = None, metric_name: str = None,
    train_size_max: int = None, train_size_min: int = None,
    test_size_max: int = None, test_size_min: int = None
):
    query_suffix = ""
    if train_size_max:
        query_suffix += f" AND ex.train_size <= {train_size_max}"
    if train_size_min:
        query_suffix += f" AND ex.train_size >= {train_size_min}"
    if test_size_max:
        query_suffix += f" AND ex.test_size <= {test_size_max}"
    if test_size_min:
        query_suffix += f" AND ex.test_size >= {test_size_min}"
    if evaluation_type:
        query_suffix += f" AND ev.evaluation_type = '{evaluation_type}'"
    if metric_name:
        query_suffix += f" AND ev.metric_name = '{metric_name}'"
    return query_suffix


def get_idx_positions_from_idx_values(idx_values: list[int], data: pd.DataFrame):
    """
    Get the index positions of the given index values in the DataFrame index.
    Args:
        idx_values (list[int]): The index values to find positions for.
        data (pd.DataFrame): The DataFrame to search in.
    Returns:
        list[int]: The index positions of the given index values.
    """

    for value in idx_values:
        if value not in data.index:
            raise ValueError(f"Value {value} not found in DataFrame index.")

    return [data.index.get_loc(value) for value in idx_values]