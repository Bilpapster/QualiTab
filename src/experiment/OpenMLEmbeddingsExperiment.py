import json
import time
import numpy as np

from . import EmbeddingsExperiment, OpenMLExperiment
from src.config import get_adaptive_inference_limit


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
            n_folds: int = 1
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
            datasets_to_skip=datasets_to_skip
        )
        self.error_type = None
        self.corrupted_columns = []
        self.corrupted_rows = []

    def run_one_experiment(self, benchmark_config=None):
        self.log(f"Running experiment with random seed: {self.random_seed}")

        self.nest_prefix()
        for experiment_mode in self.modes_iterator(benchmark_config):
            self.log(f"Running experiment with mode: {experiment_mode}")
            start_time = time.time()
            self._pollute_data_based_on_mode(experiment_mode)

            try:
                classes = list(set(self.y_train))
                limit = get_adaptive_inference_limit(
                    len(self.X_train), len(self.X_train.columns), len(classes)
                )
                test_embeddings = None
                self.nest_prefix()
                for i in range(0, len(self.X_test), limit):
                    batch = self.X_test[i:i + limit]
                    # batch_embeddings = self.model.get_embeddings(
                    #                    self.X_train,
                    #                    self.y_train,
                    #                    self.X_test,
                    #                    data_source="test",
                    # )
                    batch_embeddings = np.random.randn(1, len(batch), 192)
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
                'dataset': f"OpenML-{self.task.dataset_id}-{len(classes)}",
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'number_of_classes': len(classes),
                'number_of_features': len(self.X_train.columns),
                'random_seed': self.random_seed,
                'test_embeddings': json.dumps(test_embeddings.tolist()),
                'error_type': self.error_type,
                'corrupted_columns': json.dumps(self.corrupted_columns),
                'corrupted_rows': json.dumps(self.corrupted_rows),
                'execution_time': time.time() - start_time,
                'tag': experiment_mode.value
            }
            self.log(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
            self.write_experiment_result_to_db(result)
            self._rollback_data_based_on_mode(experiment_mode)
        self.unnest_prefix()
