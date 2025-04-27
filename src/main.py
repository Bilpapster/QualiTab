from experiment import CleanML_classification_experiment, OpenML_classification_experiment, OpenMLClassificationExperimentImproved
from config import openML_dataset_configs
from utils import get_seeds_from_env_or_else_default


if __name__ == "__main__":
    # CleanML_classification_experiment().run()
    # OpenML_classification_experiment().run()
    OpenMLClassificationExperimentImproved(
        benchmark_configs=openML_dataset_configs,
        random_seeds=get_seeds_from_env_or_else_default()
    ).run()
