import json
import time

from corruption import CorruptionsManager, CorruptionType
from .EmbeddingsExperiment import EmbeddingsExperiment
from .ExperimentMode import ExperimentMode
from .OpenMLExperiment import OpenMLExperiment
from config import get_adaptive_inference_limit
from utils import connect_to_db, get_GPU_information


class OpenMLEmbeddingsExperiment(EmbeddingsExperiment, OpenMLExperiment):
    """
    Class to run embeddings extraction experiments with OpenML datasets.
    Inherits from EmbeddingsExperiment and OpenMLExperiment.
    """
    def __init__(
            self,
            benchmark_configs: dict = None,
            random_seeds: list = None,
            datasets_to_skip: list | set = [],
            n_folds: int = 1,
            corruptions_manager=CorruptionsManager(),
            debug=False
    ):
        """
        Initializes the OpenMLClassificationExperimentImproved class.
        Args:
            benchmark_configs (dict): Configuration for the benchmark datasets.
            random_seeds (list): List of random seeds to use for experiments.
        """
        EmbeddingsExperiment.__init__(self, n_folds=n_folds)
        OpenMLExperiment.__init__(
            self,
            benchmark_configs=benchmark_configs,
            random_seeds=random_seeds,
            datasets_to_skip=datasets_to_skip,
            corruptions_manager=corruptions_manager,
        )
        self.error_type = None
        self.corrupted_columns = []
        self.corrupted_rows = []
        self.debug = debug

    def does_experiment_exist_in_db(self):
        conn, cursor = connect_to_db()
        cursor.execute("""
                       SELECT dataset_name
                       FROM embeddings_experiments
                       WHERE dataset_name = %s
                        AND random_seed = %s
                        AND error_type = %s
                        AND row_corruption_percent = %s
                        AND column_corruption_percent = %s
                        AND tag = %s
                       """,
                       (
                           self.get_dataset_name_from_dataset_id(self.task.dataset_id),
                           self.random_seed,
                           self.corruption.value,
                           self.row_corruption_percent,
                           self.column_corruption_percent,
                           self.experiment_mode.value
                       )
        )
        result = cursor.fetchall()
        return len(result) > 0

    def run_one_experiment(self, benchmark_config=None):
        import numpy as np

        # if the experiment exists already in the database, skip it
        if self.does_experiment_exist_in_db():
            self.log(f"Experiment already exists in the database. Skipping this experiment.")
            return

        np.random.seed(self.random_seed)
        self._corrupt_data_based_on_mode()

        # If the dataset was not corrupted or the corruption was not applied, skip the experiment
        if not self.experiment_mode.is_meaningful(self.corrupted_rows, self.corrupted_columns):
            self.log(f"Experiment mode {self.experiment_mode} is not meaningful.")
            self.log(f"Corrupted rows: {self.corrupted_rows}")
            self.log(f"Corrupted columns: {self.corrupted_columns}")
            self.log(f"Skipping this experiment.")
            return

        start_time = time.time()

        try:
            classes = list(set(self.y_train))
            limit = get_adaptive_inference_limit(
                len(self.X_train), len(self.X_train.columns), len(classes)
            )
            test_embeddings = None
            self.nest_prefix()
            for i in range(0, len(self.X_test), limit):
                batch = self.X_test[i:i + limit]
                if self.debug:
                    self.log(f"Running in debug mode. Generating random embeddings.")
                    batch_embeddings = np.random.randn(1, len(batch), 192)
                else:
                    batch_embeddings = self.model.get_embeddings(
                                       self.X_train,
                                       self.y_train,
                                       self.X_test,
                                       data_source="test",
                    )
                # (n_estimators, n_samples, emb_dimension) -> (n_samples, emb_dimension)
                batch_embeddings = batch_embeddings[0]

                if test_embeddings is None:
                    test_embeddings = np.array(batch_embeddings)
                else:
                    test_embeddings = np.concatenate((test_embeddings, batch_embeddings), axis=0)
                self.log(f"Finished embeddings extraction for samples {i} -- {i + len(batch) - 1}")
            self.unnest_prefix()

        except Exception as e:
            self.log(f"Error extracting embeddings for dataset {self.task.dataset_id}: {e}", "error")
            self.log(f"Skipping this experiment.", "error")
            return

        self.log(f"Embeddings extraction finished.")

        # Evaluate the model and return the results as a dictionary
        result = {
            'dataset': self.get_dataset_name_from_dataset_id(self.task.dataset_id),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'number_of_classes': len(classes),
            'number_of_features': len(self.X_train.columns),
            'random_seed': self.random_seed,
            'test_embeddings': json.dumps(test_embeddings.tolist()),
            'error_type': self.corruption.value,
            'corrupted_columns': json.dumps(self.corrupted_columns),
            'corrupted_rows': json.dumps(self.corrupted_rows),
            'execution_time': time.time() - start_time,
            'tag': self.experiment_mode.value,
            'row_corruption_percent': self.row_corruption_percent,
            'column_corruption_percent': self.column_corruption_percent,
            'gpu_info': json.dumps(get_GPU_information()),
        }
        self.log(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
        self.write_experiment_result_to_db(result)
        self._rollback_data_based_on_mode(self.experiment_mode)
