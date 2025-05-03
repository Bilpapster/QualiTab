import openml
import random
from abc import ABC, abstractmethod

from .Experiment import Experiment
from corruption import CorruptionsManager


class OpenMLExperiment(Experiment, ABC):
    def __init__(
            self,
            benchmark_configs: dict = None,
            random_seeds: list = None,
            datasets_to_skip: list | set = [],
            corruptions_manager=CorruptionsManager()
    ):
        super().__init__()
        self.benchmark_configs = benchmark_configs
        self.random_seeds = random_seeds
        self.experiment_mode = None
        self.corruption = None
        self.corruption_row_percent = None
        self.corruption_column_percent = None


        self.datasets_to_skip = set(datasets_to_skip)
        self.finished_experiments = self.get_finished_experiments()
        self.corruptions_manager = corruptions_manager
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
        """
        The big picture of the following code is:
        1. For each benchmark configuration:
            2. Get the tasks (datasets) from the benchmark configuration.
            3. For each task:
                4. For each random seed:
                    5. Load the dataset using the task and random seed.
                    6. For each experiment mode:
                        7. For each (compatible) corruption:
                            8. For each (compatible) row corruption percent:
                                9. For each (compatible) column corruption percent:
                                    10. Run one experiment with the current configuration.
        """
        self.nest_prefix() # 1. For each benchmark configuration
        for benchmark_index, benchmark_config in enumerate(self.benchmark_configs):
            # 2. Get the tasks (datasets) from the benchmark configuration.
            tasks = openml.study.get_suite(benchmark_config["name"]).tasks
            random.shuffle(tasks) # shuffle tasks to overcome bottlenecks of GPU/CPU usage

            self.log(f"Working on benchmark {benchmark_config.get('description', 'unknown')} ({benchmark_index + 1} of {len(self.benchmark_configs)})")

            self.nest_prefix() # 3. For each task
            for task_index, dataset_id in enumerate(tasks):
                """
                IMPORTANT: `dataset_id` can sometimes be different from `task.dataset_id` (OpenML's flaw).
                We use dataset_id only to get the task object from OpenML.
                Thereafter, we use task.dataset_id wherever dataset id is needed (e.g., writing to db) for consistency.
                """

                self.task = openml.tasks.get_task(dataset_id)
                if self.task.dataset_id in self.datasets_to_skip:
                    self.log(f"Skipping dataset {self.task.dataset_id} as requested by user.")
                    continue
                self.log(f"Working on dataset {self.task.dataset_id} ({task_index + 1} of {len(tasks)})")

                self.nest_prefix() # 4. For each random seed
                for random_seed_index, random_seed in enumerate(self.random_seeds):
                    # 5. Load the dataset using the task and random seed.
                    self.log(f"Random seed set to {random_seed} ({random_seed_index + 1} of {len(self.random_seeds)})")
                    self.random_seed = random_seed
                    self.load_dataset(benchmark_config, random_state=random_seed)
                    self.model = self.get_model_from_dataset_config(benchmark_config)
                    modes_corruptions_configurations = self.get_modes_and_corruptions_configurations(benchmark_config)

                    self.nest_prefix() # 6. For each experiment mode
                    for experiment_mode_index, configurations in enumerate(modes_corruptions_configurations):
                        experiment_mode, requested_corruptions, corruption_row_percents, corruption_column_percents = configurations
                        self.log(f"Experiment mode set to {experiment_mode} "
                                 f"({experiment_mode_index + 1} of {len(modes_corruptions_configurations)})")
                        self.experiment_mode = experiment_mode
                        compatible_corruptions = experiment_mode.get_compatible_corruptions_from_candidates(requested_corruptions)

                        self.nest_prefix() # 7. For each (compatible) corruption
                        for corruption_index, corruption in enumerate(compatible_corruptions):
                            self.log(f"Corruption set to {corruption} "
                                     f"({corruption_index + 1} of {len(compatible_corruptions)})")
                            self.corruption = corruption
                            compatible_row_percents = corruption.get_compatible_corruption_percents_from_candidates(
                                corruption_row_percents)

                            self.nest_prefix() # 8. For each (compatible) row corruption percent
                            for corruption_row_percent_index, corruption_row_percent in enumerate(compatible_row_percents):
                                self.log(f"Corruption percent set to {corruption_row_percent} "
                                         f"({corruption_row_percent_index + 1} of {len(compatible_row_percents)})")
                                self.corruption_row_percent = corruption_row_percent
                                compatible_column_percents = corruption.get_compatible_corruption_percents_from_candidates(
                                    corruption_column_percents)

                                self.nest_prefix() # 9. For each (compatible) column corruption percent
                                for corruption_column_percent_index, corruption_column_percent in enumerate(compatible_column_percents):
                                    self.log(f"Corruption column percent set to {corruption_column_percent} "
                                             f"({corruption_column_percent_index + 1} of {len(corruption_column_percents)})")
                                    self.corruption_column_percent = corruption_column_percent
                                    self.run_one_experiment() # 10. Run one experiment with the current configuration.

                                self.unnest_prefix() # end of loop for column corruption percents
                            self.unnest_prefix() # end of loop for row corruption percents
                        self.unnest_prefix() # end of loop for corruptions
                    self.unnest_prefix() # end of loop for experiment modes
                self.unnest_prefix() # end of loop for random seeds
            self.unnest_prefix() # end of loop for tasks
        self.unnest_prefix() # end of loop for benchmarks

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
