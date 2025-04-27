import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from . import ClassificationExperiment
from .OpenMLExperiment import OpenMLExperiment
from src.config import ExperimentMode, get_adaptive_inference_limit


class OpenMLClassificationExperiment(ClassificationExperiment, OpenMLExperiment):
    """
    Class to run classification experiments with OpenML datasets.
    Inherits from ClassificationExperiment and OpenMLExperiment.
    """
    def __init__(
            self,
            benchmark_configs: dict = None,
            random_seeds: list = None,
            datasets_to_skip: list | set = []
    ):
        """
        Initializes the OpenMLClassificationExperimentImproved class.
        Args:
            benchmark_configs (dict): Configuration for the benchmark datasets.
            random_seeds (list): List of random seeds to use for experiments.
        """
        OpenMLExperiment.__init__(
            self,
            benchmark_configs=benchmark_configs,
            random_seeds=random_seeds,
            datasets_to_skip=datasets_to_skip
        )

    def run_one_experiment(self, benchmark_config=None):
        self.logger.info(f"{self._prefix} Running experiment with random seed: {self.random_seed}")

        self.nest_prefix()
        for experiment_mode in self.modes_iterator(benchmark_config):
            self.logger.info(f"{self._prefix} Running experiment with mode: {experiment_mode}")
            start_time = time.time()
            self._pollute_data_based_on_mode(experiment_mode)

            try:
                self.model.fit(self.X_train, self.y_train)
                classes = self.model.classes_
            except Exception as e:
                self.logger.error(f"{self._prefix} Error fitting the model for dataset {self.task.dataset_id}: {e}")
                self.logger.error(f"{self._prefix} Skipping this experiment.")
                return

            self.logger.info(f"{self._prefix} Model fitted successfully. The classes are {self.model.classes_}")
            limit = get_adaptive_inference_limit(
                len(self.X_train), len(self.X_train.columns), len(self.model.classes_)
            )
            self.logger.info(f"{self._prefix} Starting inference on {len(self.X_test)} samples in batches of {limit}.")

            prediction_probabilities = None
            self.nest_prefix()
            for i in range(0, len(self.X_test), limit):
                batch = self.X_test[i:i + limit]
                # batch_probabilities = self.model.predict_proba(batch)
                batch_probabilities = np.random.randn(len(batch), len(classes))
                batch_probabilities = np.exp(batch_probabilities) / np.sum(np.exp(batch_probabilities), axis=1, keepdims=True)

                if prediction_probabilities is None:
                    prediction_probabilities = np.array(batch_probabilities)
                else:
                    prediction_probabilities = np.concatenate((prediction_probabilities, batch_probabilities),
                                                              axis=0)
                self.logger.info(f"{self._prefix} Finished inference for samples {i} -- {i + len(batch) - 1}")
            self.unnest_prefix()

            self.logger.info(f"{self._prefix} Inference finished.")
            predictions = np.array(classes)[np.argmax(prediction_probabilities, axis=1)]
            self.logger.info(
                f"{self._prefix} Prediction probabilities successfully converted to predictions of shape {predictions.shape}.")

            # Evaluate the model and return the results as a dictionary
            result = {
                'dataset': f"OpenML-{self.task.dataset_id}-{len(classes)}",
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'used_default_split': False,
                'random_seed': self.random_seed,
                'roc_auc': roc_auc_score(self.y_test, prediction_probabilities, multi_class='ovr') if len(classes) > 2
                else roc_auc_score(self.y_test, prediction_probabilities[:, 1]),
                'accuracy': accuracy_score(self.y_test, predictions),
                'recall': recall_score(self.y_test, predictions, average="micro") if len(classes) > 2
                else recall_score(self.y_test, predictions, average='binary', pos_label=classes[1]),
                'precision': precision_score(self.y_test, predictions, average="micro") if len(classes) > 2
                else precision_score(self.y_test, predictions, average='binary', pos_label=classes[1]),
                'f1_score': f1_score(self.y_test, predictions, average="micro") if len(classes) > 2
                else f1_score(self.y_test, predictions, average='binary', pos_label=classes[1]),
                'execution_time': time.time() - start_time,
                'tag': experiment_mode.value
            }
            self.logger.info(f"{self._prefix} Dataset finished. Execution time: {result['execution_time']} seconds.")
            self.write_experiment_result_to_db(result)
            self._rollback_data_based_on_mode(experiment_mode)
        self.unnest_prefix()
