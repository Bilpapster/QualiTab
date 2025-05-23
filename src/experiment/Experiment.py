from abc import ABC, abstractmethod
import pandas as pd

from corruption import CorruptionType
from utils import configure_logging, connect_to_db
from experiment.ExperimentMode import ExperimentMode


class Experiment(ABC):
    """
    Abstract base class for experiments. Implements common functionality for loading datasets,
    preparing data, and writing results to a database.
    """
    # logger is **class** attribute on purpose.
    # You can use it in the class methods as usually: self.logger.info(...)
    # for logging with indentation, use directly self.log("message", "level"), level in {'info', 'warning', 'error'}
    logger = configure_logging()

    def __init__(self):
        self.random_seed = None  # the random seed to use when splitting or sampling the dataset
        self.test_size = None  # the size of the test set to use when splitting the dataset
        self.target_name = None  # the name of the target column in the dataset
        self.model = None # the model to use for training, inference, embeddings, or data generation

        self._prefix = '' # the prefix to use for indentation in the logs

        self.features = None  # the features of the dataset, aka X (`X_train` and `X_test` together)
        self.targets = None  # the targets of the dataset, aka y (`y_train` and `y_test` together)
        self.X_train = None  # the training features
        self.y_train = None  # the training targets
        self.X_test = None  # the test features
        self.y_test = None  # the test targets

        self.experiment_mode = None # the mode of the experiment to run (e.g., CLEAN_CLEAN, CLEAN_DIRTY, ...)
        self.corruption = None # the type of corruption to apply
        self.row_corruption_percent = None # the percentage of rows to corrupt
        self.column_corruption_percent = None # the percentage of columns to corrupt
        self.corrupted_rows = None # the rows that are corrupted
        self.corrupted_columns = None # the columns that are corrupted

    def nest_prefix(self):
        """
        Adds a prefix to the logger messages for better readability.
        Should be called right before a new loop or nested block begins
        """
        self._prefix += '- - '

    def unnest_prefix(self):
        """
        Removes the last prefix added by `nest_prefix`.
        Should be called right after a loop or nested block ends.
        """
        self._prefix = self._prefix[:-4]

    def log(self, message: str, level: str = 'info'):
        """
        Logs a message with the specified logging level.
        """
        match level:
            case 'info':
                self.logger.info(f"{self._prefix} {message}")
            case 'error':
                self.logger.error(f"{self._prefix} {message}")
            case 'warning':
                self.logger.warning(f"{self._prefix} {message}")
            case _:
                raise ValueError(f"Unknown log level: {level}. Use 'info', 'error', or 'warning'.")

    # @abstractmethod
    def insertion_query(self, result: dict) -> dict:
        """
        Returns the SQL query for inserting experiment results into the database
        in the form of a dictionary that contains the query and the values to be inserted.
        """
        pass

    # @abstractmethod
    def finished_experiments_query(self) -> str:
        """
        Returns the SQL query for fetching finished datasets from the database.
        """
        pass

    def get_finished_experiments(self) -> set | list:
        """
        Fetches the finished datasets from the database using the `finished_datasets_query`.
        Returns a set of dataset IDs.
        """
        conn, cursor = connect_to_db()
        try:
            cursor.execute(self.finished_experiments_query())
            result = cursor.fetchall()
            return {self.process_finished_experiment_result_row(row) for row in result}
        except Exception as e:
            self.log(f"{self._prefix} Error fetching finished experiments: {e}", "error")
            return set()


    def does_experiment_exist_in_db(self):
        return True

    def process_finished_experiment_result_row(self, row: tuple) -> str | int:
        """
        Rows here are expected to be a tuple with a single value (concatenated string for uniquely
        identifying the experiment). In case this logic is not applicable, you should override this method.

        The function processes the row by returning it only element (you can think of is as a flattening/unboxing)
        """
        return row[0]

    def _check_nof_features(self) -> None:
        """
        Checks the number of features in the dataset against the maximum allowed by TabPFN.
        If the number of features exceeds the maximum, it randomly samples a subset of features.
        The sampling is done using the same random seed for reproducibility.
        """
        from config import TABPFN_MAX_FEATURES

        if len(self.features.columns) <= TABPFN_MAX_FEATURES:
            """If the number of features is smaller than or equal to the maximum allowed by TabPFN, do nothing."""
            return

        """If the number of features in the dataset is larger than the maximum number of features,
        randomly sample a subset of the max allowed number of features from the original features."""
        self.log(f"{self._prefix} The features ({len(self.features.columns)}) exceed the maximum allowed ({TABPFN_MAX_FEATURES}). ")
        self.log(f"{self._prefix} Sampling {TABPFN_MAX_FEATURES} features using seed {self.random_seed}.")
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
        from config import TABPFN_MAX_CLASSES
        import random

        classes = set(self.targets)
        if len(classes) <= TABPFN_MAX_CLASSES:
            """If the number of classes is smaller than or equal to the maximum allowed for TabPFN, do nothing."""
            return

        """If the number of classes is larger than the maximum allowed for TabPFN,
        randomly sample a subset of the max allowed number of classes from the original classes."""
        self.log(f"The number of classes ({len(classes)}) exceed the maximum allowed ({TABPFN_MAX_CLASSES}). ")
        self.log(f"Sampling {TABPFN_MAX_CLASSES} classes (keeping all their samples) using seed {self.random_seed}.")
        random.seed(self.random_seed) # important for reproducibility of results
        classes_to_preserve = random.sample(sorted(classes), TABPFN_MAX_CLASSES)
        self.log(f"Selected classes: {classes_to_preserve}")
        data = self._get_total_data()
        data = data[data[self.target_name].isin(classes_to_preserve)]
        self._update_features_and_targets_from_data(data)

    def _check_nof_samples(self) -> None:
        """
        Checks the number of samples in the training dataset against the maximum allowed by TabPFN.
        If the number of samples exceeds the maximum, it randomly samples a subset of samples.
        The sampling is done using the same random seed for reproducibility.
        """
        from config import TABPFN_MAX_SAMPLES

        if len(self.X_train) <= TABPFN_MAX_SAMPLES:
            """If the number of train samples is smaller than or equal to the maximum number of samples, do nothing."""
            return

        """If the number of train samples is larger than the maximum allowed for TabPFN,
        randomly sample a subset of the max allowed number of samples from the original samples."""
        self.log(f"The training samples ({len(self.X_train)}) exceed the maximum allowed ({TABPFN_MAX_SAMPLES}).")
        self.log(f"Sampling {TABPFN_MAX_SAMPLES} samples using seed {self.random_seed}.")
        train_data = self._get_train_data().sample(n=TABPFN_MAX_SAMPLES, replace=False,
                                                   random_state=self.random_seed, axis=0)
        self._update_X_y_train_from_train_data(train_data)

    def _perform_train_test_split(self) -> None:
        """
        Splits the dataset into training and testing sets using the specified test size and random seed.
        If you plan to fit a TabPFN model, consider using `_check_nof_samples` before fitting the model.
        """
        from sklearn.model_selection import train_test_split

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
            self.log(f"Experiment for {self.get_dataset_name_from_result(result)} saved to db.")
        except Exception as e:
            conn.rollback()  # Rollback in case of error
            self.log(f"Error processing this result configuration: {result}: {e}", "error")

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

    def get_model_from_dataset_config(self, dataset_config: dict):
        """
        Returns the model name from the dataset configuration.
        """
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        return TabPFNClassifier() if dataset_config['task'] == 'classification' else TabPFNRegressor()

    # @abstractmethod
    def load_dataset(self, dataset_config: dict, **kwargs) -> tuple:
        """
        Loads the dataset based on the provided configuration.
        Returns a tuple of (X_train, y_train, X_test, y_test, used_default_split, random_seed).
        """
        pass

    @staticmethod
    def get_random_seeds() -> list:
        from utils import get_seeds_from_env_or_else_default

        return get_seeds_from_env_or_else_default()

    # @abstractmethod
    def run(self):
        """
        Main method to run the experiment.
        Loads datasets, trains models, evaluates them, and writes results to the database.
        """
        # checking first for connection to the database
        connect_to_db()
        # you can add your own code overriding this method
        # just make sure to call the super().run() method


    def get_modes_and_corruptions_configurations(self, dataset_config):
        """
        Returns the experiment modes and corruption rates for the given dataset configuration.
        If no experiment modes are specified, defaults to a list with CLEAN_CLEAN as only element.
        If no corruption percentages are specified, defaults to a list with 0% corruption as only element.
        Every element this generator yields is a tuple of (experiment_mode (ExperimentMode), corruption_percents (list)).
        """
        experiment_modes = dataset_config.get('experiment_modes', [ExperimentMode.CLEAN_CLEAN])
        corruption_row_percents = dataset_config.get('row_corruption_percents', [0])
        corruption_types = dataset_config.get('corruptions', [CorruptionType.NONE])
        corruption_column_percents = dataset_config.get('corruption_column_percents', [20])

        return [
            (experiment_mode, corruption_types, corruption_row_percents, corruption_column_percents)
                for experiment_mode in experiment_modes
                if experiment_mode in ExperimentMode
        ]

        # for experiment_mode in experiment_modes:  # loop through the experiment modes found in configurations
        #     if experiment_mode in ExperimentMode:  # emit the mode if it is valid (one of the enum values)
        #         yield experiment_mode, corruption_types, corruption_percents # todo update docstring

    def _corrupt_data_based_on_mode(self):
        """
        Pollutes the data based on the experiment mode and rest configurations (e.g., corruption type)
        that are saved in the instance variables.
        """

        if self.random_seed is None:
            raise ValueError("Random seed must be set before polluting data.")
        if self.corruption is None:
            raise ValueError("Corruption type must be set before polluting data.")
        if self.row_corruption_percent is None or self.column_corruption_percent is None:
            raise ValueError("Corruption percentages must be set before polluting data.")


        match self.experiment_mode:
            case ExperimentMode.CLEAN_CLEAN:
                # No pollution needed for clean-clean mode
                self.corrupted_rows = []
                self.corrupted_columns = []
                return
            case ExperimentMode.CLEAN_DIRTY:
                self.__X_test_original = self.X_test.copy(deep=True)
                self.X_test, self.corrupted_rows, self.corrupted_columns = self.corruption.corrupt(
                    data=self.X_test, random_seed=self.random_seed,
                    row_percent=self.row_corruption_percent, column_percent=self.column_corruption_percent
                )
                return
            case ExperimentMode.DIRTY_CLEAN:
                self.__X_train_original = self.X_train.copy(deep=True)
                self.X_train, self.corrupted_rows, self.corrupted_columns = self.corruption.corrupt(
                    data=self.X_train, random_seed=self.random_seed,
                    row_percent=self.row_corruption_percent, column_percent=self.column_corruption_percent
                )
                return
            case ExperimentMode.DIRTY_DIRTY:
                self.__X_train_original = self.X_train.copy(deep=True)
                self.__X_test_original = self.X_test.copy(deep=True)
                self.X_train, train_corrupted_rows, train_corrupted_columns = self.corruption.corrupt(
                    data=self.X_train, random_seed=self.random_seed,
                    row_percent=self.row_corruption_percent, column_percent=self.column_corruption_percent
                )
                self.X_test, test_corrupted_rows, test_corrupted_columns = self.corruption.corrupt(
                    data=self.X_test, random_seed=self.random_seed,
                    row_percent=self.row_corruption_percent, column_percent=self.column_corruption_percent
                )
                self.corrupted_rows = [train_corrupted_rows, test_corrupted_rows]
                self.corrupted_columns = [train_corrupted_columns, test_corrupted_columns]
                return
            case _:
                raise ValueError(f"Unknown experiment mode: {self.experiment_mode}. Cannot pollute data.")

    def _rollback_data_based_on_mode(self, experiment_mode: ExperimentMode):
        match experiment_mode:
            case ExperimentMode.CLEAN_CLEAN:
                # No rollback needed for clean-clean mode
                return
            case ExperimentMode.CLEAN_DIRTY:
                self.X_test = self.__X_test_original.copy(deep=True)
                del self.__X_test_original
                del self.corrupted_rows
                del self.corrupted_columns
                return
            case ExperimentMode.DIRTY_CLEAN:
                self.X_train = self.__X_train_original.copy(deep=True)
                del self.__X_train_original
                del self.corrupted_rows
                del self.corrupted_columns
                return
            case ExperimentMode.DIRTY_DIRTY:
                self.X_test = self.__X_test_original.copy(deep=True)
                del self.__X_test_original
                self.X_train = self.__X_train_original.copy(deep=True)
                del self.__X_train_original
                del self.corrupted_rows
                del self.corrupted_columns
                return
            case _:
                raise ValueError(f"Unknown experiment mode: {experiment_mode}. Cannot rollback data.")

