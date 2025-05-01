import json
import time
import numpy as np

from .EmbeddingsExperiment import EmbeddingsExperiment
from .OpenMLExperiment import OpenMLExperiment
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
            n_folds: int = 1,
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
            datasets_to_skip=datasets_to_skip
        )
        self.error_type = None
        self.corrupted_columns = []
        self.corrupted_rows = []
        self.debug = debug

    def run_one_experiment(self, benchmark_config=None):
        if self.random_seed is None or self.random_seed_index is None or self.random_seeds is None:
            raise ValueError("Random seed(s) (and their indices) must be set before running a specific experiment.")

        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data must be loaded before running a specific experiment.")

        if benchmark_config is None:
            raise ValueError("Benchmark configuration must be provided before running a specific experiment.")

        self.log(f"Running experiment with random seed {self.random_seed} ({self.random_seed_index} of {len(self.random_seeds)})")

        self.nest_prefix()
        for experiment_mode, corruption_percents in self.modes_iterator(benchmark_config):
            self.log(f"Running experiment with mode: {experiment_mode}")
        #     self.nest_prefix()
        # #######
        #     for corruption_percent_index, corruption_percent in enumerate(corruption_percents):
        #         self.log(f"Running experiment with corruption percent {corruption_percent}% ({corruption_percent_index} of {len(corruption_percents)})")
        #         self.error_type = 'missing values' # todo handle a loop with different error types
        #         self.corrupted_columns = []
        #         self.corrupted_rows = []
        #
        #         # Set the random seed for reproducibility
        #         np.random.seed(self.random_seed)
        #         self.random_seed = self.random_seeds[self.random_seed_index]
        #
        #         # Set the corruption percent
        #         if experiment_mode == ExperimentMode.CLEAN_CLEAN:
        #             self.corruption_percent = 0
        #         else:
        #             self.corruption_percent = corruption_percent
        #
        #     self.unnest_prefix()
        # self.unnest_prefix()
        # #####

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
                'error_type': self.error_type,
                'corrupted_columns': json.dumps(self.corrupted_columns),
                'corrupted_rows': json.dumps(self.corrupted_rows),
                'execution_time': time.time() - start_time,
                'tag': experiment_mode.value,
                'corruption_percent': 0, # todo: important to handle the corruption percent correctly with Jenga
            }
            self.log(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
            self.write_experiment_result_to_db(result)
            self._rollback_data_based_on_mode(experiment_mode)
        self.unnest_prefix()
