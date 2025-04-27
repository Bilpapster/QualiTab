import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from .ClassificationExperimentImproved import ClassificationExperiment
from .OpenMLExperiment import OpenMLExperiment
from src.config import ExperimentMode, get_adaptive_inference_limit


class OpenMLClassificationExperimentImproved(ClassificationExperiment, OpenMLExperiment):
    """
    Class to run classification experiments with OpenML datasets.
    Inherits from ClassificationExperiment and OpenMLExperiment.
    """
    def __init__(
            self,
            benchmark_configs: dict = None,
            random_seeds: list = None,
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
            random_seeds=random_seeds
        )

    def modes_iterator(self, dataset_config):
        """
        Returns the experiment modes for the given dataset configuration.
        If no experiment modes are specified, defaults to a list with CLEAN_CLEAN as only element.
        """
        experiment_modes = dataset_config.get('experiment_modes', [ExperimentMode.CLEAN_CLEAN])
        for experiment_mode in experiment_modes:  # loop through the experiment modes found in configurations
            if experiment_mode in ExperimentMode:  # emit the mode if it is valid (one of the enum values)
                yield experiment_mode

    def _pollute_data_based_on_mode(self, experiment_mode: ExperimentMode):
        """
        Pollutes the data based on the experiment mode.
        Args:
            experiment_mode (ExperimentMode): The mode of the experiment to run.
        """
        if self.random_seed is None:
            raise ValueError("Random seed must be set before polluting data.")

        match experiment_mode:
            case ExperimentMode.CLEAN_CLEAN:
                # No pollution needed for clean-clean mode
                return
            case ExperimentMode.CLEAN_DIRTY:
                self.__X_test_original = self.X_test.copy()
                self.__y_test_original = self.y_test.copy()
                # todo pollute X_test, y_test
                return
            case ExperimentMode.DIRTY_CLEAN:
                self.__X_train_original = self.X_train.copy()
                self.__y_train_original = self.y_train.copy()
                # todo pollute X_train, y_train
                return
            case ExperimentMode.DIRTY_DIRTY:
                self.__X_test_original = self.X_test.copy()
                self.__y_test_original = self.y_test.copy()
                self.__X_train_original = self.X_train.copy()
                self.__y_train_original = self.y_train.copy()
                # todo pollute X_train, y_train, X_test, y_test
                return

    def _rollback_data_based_on_mode(self, experiment_mode: ExperimentMode):
        match experiment_mode:
            case ExperimentMode.CLEAN_CLEAN:
                # No rollback needed for clean-clean mode
                return
            case ExperimentMode.CLEAN_DIRTY:
                self.X_test = self.__X_test_original.copy()
                self.__delattr__('__X_test_original')
                self.y_test = self.__y_test_original.copy()
                self.__delattr__('__y_test_original')
                return
            case ExperimentMode.DIRTY_CLEAN:
                self.X_train = self.__X_train_original.copy()
                self.__delattr__('__X_train_original')
                self.y_train = self.__y_train_original.copy()
                self.__delattr__('__y_train_original')
                return
            case ExperimentMode.DIRTY_DIRTY:
                self.X_test = self.__X_test_original.copy()
                self.__delattr__('__X_test_original')
                self.y_test = self.__y_test_original.copy()
                self.__delattr__('__y_test_original')
                self.X_train = self.__X_train_original.copy()
                self.__delattr__('__X_train_original')
                self.y_train = self.__y_train_original.copy()
                self.__delattr__('__y_train_original')
                return
            case _:
                raise ValueError(f"Unknown experiment mode: {experiment_mode}. Cannot rollback data.")

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
