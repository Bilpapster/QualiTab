from abc import ABC, abstractmethod
import random
import pandas as pd
import sys

sys.path.append('..')

from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.model_selection import train_test_split
from src.utils import configure_logging, connect_to_db, get_seeds_from_env_or_else_default
from src.config import TABPFN_MAX_SAMPLES, TABPFN_MAX_FEATURES, TABPFN_MAX_CLASSES


class Experiment(ABC):
    """
    Abstract base class for experiments. Implements common functionality for loading datasets,
    preparing data, and writing results to a database.
    """
    # logger is **class** attribute on purpose.
    # You can use it in the class methods as usually: self.logger.info(...)
    logger = configure_logging()

    def __init__(self):
        self.random_seed = None  # the random seed to use when splitting or sampling the dataset
        self.test_size = None  # the size of the test set to use when splitting the dataset
        self.target_name = None  # the name of the target column in the dataset
        self.model = None # the model to use for training, inference, embeddings, or data generation

        self._prefix = '--' # the prefix to use for indentation in the logs

        self.features = None  # the features of the dataset, aka X (`X_train` and `X_test` together)
        self.targets = None  # the targets of the dataset, aka y (`y_train` and `y_test` together)
        self.X_train = None  # the training features
        self.y_train = None  # the training targets
        self.X_test = None  # the test features
        self.y_test = None  # the test targets

    def nest_prefix(self):
        """
        Adds a prefix to the logger messages for better readability.
        Should be called right before a new loop or nested block begins
        """
        self._prefix += '----'

    def unnest_prefix(self):
        """
        Removes the last prefix added by `nest_prefix`.
        Should be called right after a loop or nested block ends.
        """
        self._prefix = self._prefix[:-4]

    @abstractmethod
    def insertion_query(self, result: dict) -> dict:
        """
        Returns the SQL query for inserting experiment results into the database
        in the form of a dictionary that contains the query and the values to be inserted.
        """
        pass

    def _check_nof_features(self) -> None:
        """
        Checks the number of features in the dataset against the maximum allowed by TabPFN.
        If the number of features exceeds the maximum, it randomly samples a subset of features.
        The sampling is done using the same random seed for reproducibility.
        """
        if len(self.features.columns) <= TABPFN_MAX_FEATURES:
            """If the number of features is smaller than or equal to the maximum allowed by TabPFN, do nothing."""
            return

        """If the number of features in the dataset is larger than the maximum number of features,
        randomly sample a subset of the max allowed number of features from the original features."""
        self.logger.info(
            f"{self._prefix} The features ({len(self.features.columns)}) exceed the maximum allowed ({TABPFN_MAX_FEATURES}). "
            f"{self._prefix} Sampling {TABPFN_MAX_FEATURES} features using seed {self.random_seed}."
        )
        self.features = self.features.sample(
            n=TABPFN_MAX_FEATURES, replace=False,
            random_state=self.random_seed, axis=1
        )

    def _get_total_data(self) -> pd.DataFrame:
        """
        Returns the total data (features and targets) as a DataFrame.
        Raises ValueError if features or targets are not set.
        """
        if self.features is None or self.targets is None:
            raise ValueError("Features and targets must be set before getting total data.")

        return pd.concat([self.features, self.targets], axis=1)

    def _get_train_data(self) -> pd.DataFrame:
        """
        Returns the training data (features and targets) as a DataFrame.
        Raises ValueError if `X_train` and/or `y_train` is not set.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data must be set before getting train data.")

        return pd.concat([self.X_train, self.y_train], axis=1)

    def _update_features_and_targets_from_data(self, data: pd.DataFrame) -> None:
        """
        Updates the features and targets attributes based on the provided data.
        """
        self.features = data.drop(columns=[self.target_name])
        self.targets = data[self.target_name]

    def _update_X_y_train_from_train_data(self, train_data: pd.DataFrame) -> None:
        """
        Updates the X_train and y_train attributes based on the provided train data.
        """
        self.X_train = train_data.drop(columns=[self.target_name])
        self.y_train = train_data[self.target_name]

    def _check_nof_classes(self) -> None:
        """
        Checks the number of classes in the dataset against the maximum allowed by TabPFN.
        If the number of classes exceeds the maximum, it randomly samples a subset of classes.
        The sampling is done using the same random seed for reproducibility.
        """
        classes = set(self.targets)
        if len(classes) <= TABPFN_MAX_CLASSES:
            """If the number of classes is smaller than or equal to the maximum allowed for TabPFN, do nothing."""
            return

        """If the number of classes is larger than the maximum allowed for TabPFN,
        randomly sample a subset of the max allowed number of classes from the original classes."""
        self.logger.info(
            f"{self._prefix} The number of classes ({len(classes)}) exceed the maximum allowed ({TABPFN_MAX_CLASSES}). "
            f"{self._prefix} Sampling {TABPFN_MAX_CLASSES} classes (keeping all their samples) using seed {self.random_seed}."
        )
        classes_to_preserve = random.sample(classes, TABPFN_MAX_CLASSES)
        self.logger.info(f"{self._prefix} Selected classes: {classes_to_preserve}")
        data = self._get_total_data()
        data = data[data[self.target_name].isin(classes_to_preserve)]
        self._update_features_and_targets_from_data(data)

    def _check_nof_samples(self) -> None:
        """
        Checks the number of samples in the training dataset against the maximum allowed by TabPFN.
        If the number of samples exceeds the maximum, it randomly samples a subset of samples.
        The sampling is done using the same random seed for reproducibility.
        """
        if len(self.X_train) <= TABPFN_MAX_SAMPLES:
            """If the number of train samples is smaller than or equal to the maximum number of samples, do nothing."""
            return

        """If the number of train samples is larger than the maximum allowed for TabPFN,
        randomly sample a subset of the max allowed number of samples from the original samples."""
        self.logger.info(
            f"{self._prefix} The training samples ({len(self.X_train)}) exceed the maximum allowed ({TABPFN_MAX_SAMPLES}). "
            f"{self._prefix} Sampling {TABPFN_MAX_SAMPLES} samples using seed {self.random_seed}."
        )
        train_data = self._get_train_data().sample(n=TABPFN_MAX_SAMPLES, replace=False,
                                                   random_state=self.random_seed, axis=0)
        self._update_X_y_train_from_train_data(train_data)

    def _perform_train_test_split(self) -> None:
        """
        Splits the dataset into training and testing sets using the specified test size and random seed.
        If you plan to fit a TabPFN model, consider using `_check_nof_samples` before fitting the model.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.targets,
            test_size=self.test_size, random_state=self.random_seed
        )

    def prepare_data(self):
        """
        Prepares the data for training and testing.

        Raises:
            ValueError: If features, targets, random seed, test size, or target name are not set.
        """
        if self.features is None or self.targets is None:
            raise ValueError("Features and targets must be set before preparing data.")

        if self.random_seed is None:
            raise ValueError(
                "Random seed must be set before preparing data. "
                "Setting the random seed ensures reproducibility of the results."
            )

        if self.test_size is None or self.target_name is None:
            raise ValueError("The test size and the name of the target column must be set before preparing data.")

        self._check_nof_features()
        self._check_nof_classes()
        self._perform_train_test_split()
        self._check_nof_samples()  # check made after the split to avoid data leakage

    def write_experiment_result_to_db(self, result: dict) -> None:
        """
        Writes the experiment result to the database.
        It connects to the PostgreSQL database using environment variables,
        inserts the experiment data into the appropriate table,
        and logs the success or failure of the operation.
        """
        conn, cursor = connect_to_db()
        try:
            cursor.execute(**self.insertion_query(result))
            conn.commit()
            self.logger.info(f"{self._prefix} Experiment for {self.get_dataset_name_from_result(result)} saved to db.")
        except Exception as e:
            conn.rollback()  # Rollback in case of error
            self.logger.error(f"{self._prefix} Error processing this result configuration: {result}: {e}")

    @staticmethod
    def get_dataset_name_from_result(result: dict) -> str:
        """
        Returns the dataset name from the result dictionary.
        """
        return result['dataset']

    @staticmethod
    def get_dataset_name_from_config(dataset_config: dict) -> str:
        """
        Returns the dataset name from the dataset configuration.
        """
        return dataset_config['name']

    @staticmethod
    def get_model_from_dataset_config(dataset_config: dict):
        """
        Returns the model name from the dataset configuration.
        """
        return TabPFNClassifier() if dataset_config['task'] == 'classification' else TabPFNRegressor()

    @abstractmethod
    def load_dataset(self, dataset_config: dict, **kwargs) -> tuple:
        """
        Loads the dataset based on the provided configuration.
        Returns a tuple of (X_train, y_train, X_test, y_test, used_default_split, random_seed).
        """
        pass

    @staticmethod
    def get_random_seeds() -> list:
        return get_seeds_from_env_or_else_default()

    @abstractmethod
    def run(self):
        """
        Main method to run the experiment.
        Loads datasets, trains models, evaluates them, and writes results to the database.
        """
        # checking first for connection to the database
        connect_to_db()
        # you can add your own code overriding this method
        # just make sure to call the super().run() method
