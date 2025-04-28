from abc import ABC

from . import Experiment
from tabpfn import TabPFNClassifier
import uuid

class ClassificationExperiment(Experiment, ABC):
    def insertion_query(self, result: dict) -> dict:
        return {
            'query': """
             INSERT INTO classification_experiments (
                 experiment_id,
                 dataset_name,
                 train_size,
                 test_size,
                 used_default_split,
                 random_seed,
                 roc_auc,
                 accuracy,
                 recall,
                 precision,
                 f1_score,
                 execution_time,
                 tag
             )
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                     """,
            'vars': (
                str(uuid.uuid4()),
                result['dataset'],
                result['train_size'],
                result['test_size'],
                result['used_default_split'],
                result['random_seed'],
                result['roc_auc'],
                result['accuracy'],
                result['recall'],
                result['precision'],
                result['f1_score'],
                result['execution_time'],
                result['tag'],
            ),
        }

    def finished_datasets_query(self) -> str:
        return """
               SELECT DISTINCT dataset_name 
               FROM classification_experiments 
               GROUP BY dataset_name 
               HAVING COUNT(*) >= 10
        """

    def get_model_from_dataset_config(self, dataset_config: dict):
        return TabPFNClassifier()
