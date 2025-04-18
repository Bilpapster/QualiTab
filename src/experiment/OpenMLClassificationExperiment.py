from .ClassificationExperiment import ClassificationExperiment


class CleanML_classification_experiment(ClassificationExperiment):
    def load_dataset(self, dataset_config: dict, **kwargs) -> tuple:
        from src.load_cleanML_dataset import load_dataset

        return load_dataset(dataset_config, mode='force_manual_split', random_seed=kwargs['random_seed'])

    def get_dataset_configs(self) -> list:
        import sys
        sys.path.append('..')
        from src.config import cleanML_dataset_configs

        return cleanML_dataset_configs

