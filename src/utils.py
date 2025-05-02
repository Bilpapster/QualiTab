import logging
from dotenv import load_dotenv
import openml

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
