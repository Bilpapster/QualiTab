import sys
import logging
from experiment import OpenMLEmbeddingsExperiment
from config import openML_dataset_configs
from utils import get_seeds_from_env_or_else_default

logging.getLogger('openml').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

if __name__ == "__main__":
    OpenMLEmbeddingsExperiment(
        benchmark_configs=openML_dataset_configs[:1], # [:1] to use only OpenML-CC18
        random_seeds=get_seeds_from_env_or_else_default(),
        debug=True if "debug" in sys.argv else False,
    ).run()
