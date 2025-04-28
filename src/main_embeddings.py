from experiment import OpenMLClassificationExperiment, OpenMLEmbeddingsExperiment
from config import openML_dataset_configs
from utils import (
    get_seeds_from_env_or_else_default,
    get_finished_datasets_from_env_or_else_empty
)
import sys


if __name__ == "__main__":
    # OpenMLClassificationExperiment(
    #     benchmark_configs=openML_dataset_configs,
    #     random_seeds=get_seeds_from_env_or_else_default(),
    #     datasets_to_skip=get_finished_datasets_from_env_or_else_empty(),
    # ).run()
    # print(openML_dataset_configs[:1])
    debug_mode = True if "debug" in sys.argv else False
    print(f"Debug mode: {debug_mode}")
    OpenMLEmbeddingsExperiment(
        benchmark_configs=openML_dataset_configs[:1], # [:1] to use only OpenML-CC18
        random_seeds=get_seeds_from_env_or_else_default(),
        datasets_to_skip=get_finished_datasets_from_env_or_else_empty(),
        debug=True if "debug" in sys.argv else False,
    ).run()
