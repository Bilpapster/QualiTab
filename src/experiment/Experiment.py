from abc import ABC, abstractmethod
from tabpfn import TabPFNClassifier, TabPFNRegressor

from utils import configure_logging, connect_to_db


class Experiment(ABC):
    @abstractmethod
    def insertion_query(self, result: dict) -> dict:
        """
        Returns the SQL query for inserting experiment results into the database
        in the form of a dictionary that contains the query and the values to be inserted.
        """
        pass

    def write_experiment_result_to_db(self, result: dict) -> None:
        """
        Writes the experiment result to the database.
        It connects to the PostgreSQL database using environment variables,
        inserts the experiment data into the appropriate table,
        and logs the success or failure of the operation.
        """
        conn, cursor = connect_to_db()
        logger = configure_logging()
        try:
            cursor.execute(**self.insertion_query(result))
            conn.commit()
            logger.info(f"Experiment for {self.get_dataset_name_from_result(result)} saved to db.")
        except Exception as e:
            conn.rollback()  # Rollback in case of error
            logger.error(f"Error processing this result configuration: {result}: {e}")

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

    @abstractmethod
    def get_dataset_configs(self) -> list:
        """
        Returns a list of dataset configurations.
        """
        pass

    @staticmethod
    def get_random_seeds() -> list:
        from utils import get_seeds_from_env_or_else_default
        return get_seeds_from_env_or_else_default()

    @abstractmethod
    def run(self):
        """
        Main method to run the experiment.
        Loads datasets, trains models, evaluates them, and writes results to the database.
        """
        pass
