from abc import ABC, abstractmethod
import openml

from .Experiment import Experiment


class OpenMLExperiment(Experiment, ABC):
    def __init__(
            self,
            benchmark_configurations: dict = None,
            random_seeds: list = None,
    ):
        super().__init__()
        self.benchmark_configurations = benchmark_configurations
        self.random_seeds = random_seeds
        self.task = None

    def load_dataset(self, dataset_config: dict, **kwargs) -> None:
        """
        Load the dataset from OpenML using `self.task`.
        After execution, the instance variables `features` and `targets` will be set to the
        corresponding dataframes. Also, `X_train`, `y_train`, `X_test`, `y_test` will be set
        and ready for use.
        Raises:
            ValueError: If the task is not set before loading the dataset.
        """
        if self.task is None:
            raise ValueError("(OpenML) task must be set before loading the dataset.")

        self.target_name = self.task.target_name
        self.features, self.targets = self.task.get_X_and_y(dataset_format='dataframe')
        self.prepare_data()

    def run(self):
        if self.benchmark_configurations is None:
            raise ValueError("Benchmark configurations must be provided before running experiments.")

        if self.random_seeds is None:
            raise ValueError("Random seeds must be provided before running experiments.")

        super().run() # makes sure any preparatory checks (e.g., connection to db) are done

        for benchmark_config in self.benchmark_configurations:
            benchmark_suite = openml.study.get_suite(benchmark_config["name"])
            tasks = benchmark_suite.tasks
            datasets_to_skip = benchmark_config["datasets_to_skip"]

            for task_index, dataset_id in enumerate(tasks):
                # important dataset_id can be different from task.dataset_id
                # using dataset_id only to get task and task.dataset_id thereafter for consistency
                self.task = openml.tasks.get_task(dataset_id)
                if self.task.dataset_id in datasets_to_skip:
                    continue

                for random_seed in self.random_seeds:
                    self.random_seed = random_seed
                    self.load_dataset(dataset_id, random_state=random_seed)
                    self.run_one_experiment(benchmark_config)

    @abstractmethod
    def run_one_experiment(self, benchmark_config=None):
        # random seed and task can be found in instance attributes
        pass
