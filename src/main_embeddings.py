import sys
import logging
from experiment.OpenMLEmbeddingsExperiment import OpenMLEmbeddingsExperiment
from corruption import Corruption, CorruptionType, CorruptionsManager
# from corruption.CorruptionsManager import CorruptionsManager
# from corruption.CorruptionType import CorruptionType
from config import openML_dataset_configs
from utils import get_seeds_from_env_or_else_default, get_datasets_to_skip_from_env_or_else_empty

logging.getLogger('openml').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

mcar_corruption = Corruption(type=CorruptionType.MCAR, name='MCAR', description="Missing Values Completely at Random")

scar_corruption = Corruption(type=CorruptionType.SCAR, name='SCAR', description="Scaling Completely at Random")

cscar_corruption = Corruption(type=CorruptionType.CSCAR, name='CSCAR', description="Categorical Shift Completely at Random")

corruptions_manager = CorruptionsManager() \
    .add_corruption(mcar_corruption) \
    .add_corruption(scar_corruption) \
    .add_corruption(cscar_corruption)

if __name__ == "__main__":
    OpenMLEmbeddingsExperiment(
        benchmark_configs=openML_dataset_configs[:1],  # [:1] to use only OpenML-CC18
        random_seeds=get_seeds_from_env_or_else_default(),
        datasets_to_skip=get_datasets_to_skip_from_env_or_else_empty(),
        debug=True if "debug" in sys.argv else False,
    ).run()
