import logging
from experiment import OpenMLClassificationExperiment
from config import openML_dataset_configs
from utils import get_seeds_from_env_or_else_default, get_datasets_to_skip_from_env_or_else_empty

logging.getLogger('openml').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

if __name__ == "__main__":
    OpenMLClassificationExperiment(
        benchmark_configs=openML_dataset_configs,
        random_seeds=get_seeds_from_env_or_else_default(),
        datasets_to_skip=get_datasets_to_skip_from_env_or_else_empty(),
    ).run()
