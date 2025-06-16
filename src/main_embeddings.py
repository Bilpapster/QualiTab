import logging
from experiment.OpenMLEmbeddingsExperiment import OpenMLEmbeddingsExperiment
from config import openML_dataset_configs
from utils import (
    get_seeds_from_env_or_else_default,
    get_datasets_to_skip_from_env_or_else_empty,
    get_debug_mode_from_env_or_else_false
)

logging.getLogger('openml').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

if __name__ == "__main__":
    OpenMLEmbeddingsExperiment(
        benchmark_configs=openML_dataset_configs[:1],  # [:1] to use only OpenML-CC18
        random_seeds=get_seeds_from_env_or_else_default(),
        datasets_to_skip=get_datasets_to_skip_from_env_or_else_empty(),
        debug=get_debug_mode_from_env_or_else_false(),
    ).run()
