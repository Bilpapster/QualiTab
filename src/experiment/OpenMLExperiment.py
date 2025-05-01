import openml
import random
from abc import ABC, abstractmethod

from .Experiment import Experiment


class OpenMLExperiment(Experiment, ABC):
    def __init__(
            self,
            benchmark_configs: dict = None,
            random_seeds: list = None,
            datasets_to_skip: list | set = []
    ):
        super().__init__()
        self.benchmark_configs = benchmark_configs
        self.random_seeds = random_seeds
        self.random_seed_index = None
        self.datasets_to_skip = set(datasets_to_skip)
        self.finished_experiments = self.get_finished_experiments()
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
        self.target_name = self.task.target_name
        self.test_size = dataset_config.get("test_size", 0.3)
        self.prepare_data()

    def run(self):
        if self.benchmark_configs is None:
            raise ValueError("Benchmark configurations must be provided before running experiments.")

        if self.random_seeds is None:
            raise ValueError("Random seeds must be provided before running experiments.")

        super().run() # makes sure any preparatory checks (e.g., connection to db) are done

        for benchmark_config in self.benchmark_configs:
            benchmark_suite = openml.study.get_suite(benchmark_config["name"])
            tasks = benchmark_suite.tasks
            random.shuffle(tasks) # shuffle tasks to overcome bottlenecks of GPU/CPU usage

            self.log(f"Working on benchmark {benchmark_config.get('description', 'unknown')}")

            self.nest_prefix()
            for task_index, dataset_id in enumerate(tasks):
                # IMPORTANT: dataset_id can sometimes be different from task.dataset_id (OpenML's flaw).
                # We use dataset_id only to get the task object from OpenML.
                # Thereafter, we use task.dataset_id wherever dataset id is needed for consistency.
                self.task = openml.tasks.get_task(dataset_id)
                if self.task.dataset_id in self.datasets_to_skip:
                    self.log(f"Skipping dataset {self.task.dataset_id} as requested by user.")
                    continue

                # Important: we no longer skip already processed datasets to accommodate for more granular skipping.
                # For example, you may do not want to skip all experiments for a dataset, but only some of them:
                # e.g., you may want to skip all experiments with random seed 0, but not the others.
                # You should handle the skipping of already finished experiments in the `run_one_experiment` method.

                self.log(f"Working on dataset {self.task.dataset_id} ({task_index + 1}/{len(tasks)})")

                self.nest_prefix()
                for random_seed_index, random_seed in enumerate(self.random_seeds):
                    self.random_seed_index = random_seed_index
                    self.random_seed = random_seed
                    self.load_dataset(benchmark_config, random_state=random_seed)
                    self.model = self.get_model_from_dataset_config(benchmark_config)
                    self.run_one_experiment(benchmark_config)
                self.unnest_prefix()
            self.unnest_prefix()

    def get_dataset_name_from_dataset_id(self, dataset_id: int) -> str:
        """
        Get the dataset name from the dataset id. Currently, prepends 'OpenML-' to the dataset id.
        For example, if the dataset id is 123, the name will be 'OpenML-123'.
        Args:
            dataset_id (int): The dataset id.
        Returns:
            str: The dataset name.
        """
        return f"OpenML-{dataset_id}"

    @abstractmethod
    def run_one_experiment(self, benchmark_config=None):
        # random seed, data split to X/y train/test and model can be found in instance attributes
        pass
