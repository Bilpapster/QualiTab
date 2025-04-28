from abc import ABC

from . import Experiment
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding
import uuid

class EmbeddingsExperiment(Experiment, ABC):
    def __init__(self, n_folds: int = 0):
        super().__init__()
        self.n_folds = n_folds

    def insertion_query(self, result: dict) -> dict:
        return {
            'query': """
             INSERT INTO embeddings_experiments (
                 experiment_id,
                 dataset_name,
                 train_size,
                 test_size,
                 number_of_classes,
                 number_of_features,
                 random_seed,
                 test_embeddings,
                 error_type,
                 corrupted_columns,
                 corrupted_rows,
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
                result['number_of_classes'],
                result['number_of_features'],
                result['random_seed'],
                result['test_embeddings'],
                result['error_type'],
                result['corrupted_columns'],
                result['corrupted_rows'],
                result['execution_time'],
                result['tag'],
            ),
        }

    def get_model_from_dataset_config(self, dataset_config: dict):
        if dataset_config['task'] == 'classification':
            clf = TabPFNClassifier(n_estimators=1, random_state=self.random_seed)
            return TabPFNEmbedding(tabpfn_clf=clf) # todo in future version we can also use n_folds

        # if the task is other than classification, we fall back to regression
        reg = TabPFNRegressor(n_estimators=1, random_state=self.random_seed)
        return TabPFNEmbedding(tabpfn_reg=reg) # todo in future version we can also use n_folds
