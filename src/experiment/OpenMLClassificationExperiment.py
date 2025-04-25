from .ClassificationExperiment import ClassificationExperiment
import openml, sys, pandas as pd, numpy as np, time, random
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

sys.path.append("..")
from src.utils import configure_logging, get_seeds_from_env_or_else_default, get_finished_datasets_from_env_or_else_empty
from src.config import TABPFN_MAX_SAMPLES, TABPFN_MAX_FEATURES, TABPFN_MAX_CLASSES, get_adaptive_inference_limit
from sklearn.model_selection import train_test_split


class OpenML_classification_experiment(ClassificationExperiment):
    def get_dataset_configs(self) -> list:
        import sys
        sys.path.append('..')
        from src.config import openML_dataset_configs

        return openML_dataset_configs

    def load_dataset(self, dataset_config: dict, **kwargs) -> tuple:
        task = dataset_config  # dataset config here is not a dictionary but an OpenML task
        random_seed = kwargs.get('random_seed', 42)
        logger = configure_logging()

        target_column = task.target_name

        features, targets = task.get_X_and_y(dataset_format='dataframe')
        if len(features.columns) > TABPFN_MAX_FEATURES:
            """If the number of features in the dataset is larger than the maximum number of features,
            randomly sample a subset of the max allowed number of features from the original features.
            Important: The same random seed is used for reproducibility."""
            logger.info(
                f"The features ({len(features.columns)}) exceed the maximum allowed ({TABPFN_MAX_FEATURES}). "
                f"Sampling {TABPFN_MAX_FEATURES} features using seed {random_seed}."
            )
            features = features.sample(n=TABPFN_MAX_FEATURES, replace=False, random_state=random_seed, axis=1)

        data = pd.concat([features, targets], axis=1)
        data.replace('__MISSING__', np.nan, inplace=True)
        # data = data.rename(columns=lambda x: str(x).replace('_',''))
        # data.replace('__MISSING__', np.nan, inplace=True)

        classes = set(targets)
        if len(classes) > TABPFN_MAX_CLASSES:
            """If the number of classes is larger than the maximum allowed for TabPFN,
            randomly sample a subset of the max allowed number of classes from the original classes.
            Important: The same random seed is used for reproducibility."""
            logger.info(
                f"The number of classes ({len(classes)}) exceed the maximum allowed ({TABPFN_MAX_CLASSES}). "
                f"Sampling {TABPFN_MAX_CLASSES} classes (keeping all their samples) using seed {random_seed}."
            )
            random.seed(random_seed)
            classes_to_preserve = random.sample(classes, TABPFN_MAX_CLASSES)
            logger.info(f"Selected classes: {classes_to_preserve}")
            try:
                data = data[data[target_column].isin(classes_to_preserve)]
            except Exception as e:
                logger.error(f"Error filtering classes for dataset {task.dataset_id}: {e}")
                pass


        train_data, test_data = train_test_split(data, test_size=0.3, random_state=random_seed)

        if len(train_data) > TABPFN_MAX_SAMPLES:
            """If the number of train samples is larger than the maximum number of samples,
            randomly sample a subset of the max allowed number of samples from the original samples.
            Important: The same random seed is used for reproducibility."""
            logger.info(
                f"The training samples ({len(train_data)}) exceed the maximum allowed ({TABPFN_MAX_SAMPLES}). "
                f"Sampling {TABPFN_MAX_SAMPLES} samples using seed {random_seed}."
            )
            train_data = train_data.sample(n=TABPFN_MAX_SAMPLES, replace=False,
                                           random_state=random_seed, axis=0)

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        return X_train, y_train, X_test, y_test, False, random_seed

    def run(self):
        logger = configure_logging()
        seeds = get_seeds_from_env_or_else_default()
        finished_datasets = get_finished_datasets_from_env_or_else_empty()

        benchmark_configs = self.get_dataset_configs()
        for benchmark_config in benchmark_configs:
            if benchmark_config['task'] != 'classification':
                continue
            benchmark_suite = openml.study.get_suite(benchmark_config['name'])
            logger.info(f"Running experiment for all datasets in the benchmark suite {benchmark_suite.alias}")
            tasks = benchmark_suite.tasks
            i = 0
            for task_index, dataset_id in enumerate(tasks):
                task = openml.tasks.get_task(dataset_id)
                logger.info(f"---- DATASET ID {task.dataset_id} ({task_index} of {len(tasks)}) ----")
                if task.dataset_id in finished_datasets:
                    logger.info(f"Dataset {task.dataset_id} is already processed. Skipping.")
                    continue

                for seed_index, random_seed in enumerate(seeds):
                    logger.info(f"---- SEED {random_seed} ({seed_index} of {len(seeds)}) ----")
                    start_time = time.time()

                    X_train, y_train, X_test, y_test, _, _ = self.load_dataset(task, random_seed=random_seed)

                    clf = self.get_model_from_dataset_config(benchmark_config)
                    try:
                        clf.fit(X_train, y_train)
                    except Exception as e:
                        logger.error(f"Error fitting the model for dataset {task.dataset_id}: {e}")
                        continue
                    classes = clf.classes_
                    logger.info(f"TabPFN classifier fitted. The target classes are {classes}")

                    LIMIT = get_adaptive_inference_limit(len(X_train), len(X_train.columns), len(classes))
                    logger.info(f"Starting inference on {len(X_test)} samples in batches of {LIMIT}.")

                    prediction_probabilities = None
                    for i in range(0, len(X_test), LIMIT):
                        batch = X_test[i:i + LIMIT]
                        batch_probabilities = clf.predict_proba(batch)
                        # batch_probabilities = np.random.randn(len(batch), len(classes))
                        # batch_probabilities = np.exp(batch_probabilities) / np.sum(np.exp(batch_probabilities), axis=1, keepdims=True)

                        if prediction_probabilities is None:
                            prediction_probabilities = np.array(batch_probabilities)
                        else:
                            prediction_probabilities = np.concatenate((prediction_probabilities, batch_probabilities),
                                                                      axis=0)
                        logger.info(f"Finished inference for samples {i} -- {i + len(batch) - 1}")
                    logger.info(f"Inference finished.")
                    predictions = np.array(classes)[np.argmax(prediction_probabilities, axis=1)]
                    logger.info(
                        f"Prediction probabilities successfully converted to predictions of shape {predictions.shape}.")

                    # Evaluate the model and return the results as a dictionary
                    result = {
                        'dataset': f"OpenML-{task.dataset_id}-{len(classes)}",
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'used_default_split': False,
                        'random_seed': random_seed,
                        # Caution: ROC AUC is valid this way only for binary classification!
                        'roc_auc': roc_auc_score(y_test, prediction_probabilities, multi_class='ovr') if len(
                            classes) > 2 else roc_auc_score(y_test, prediction_probabilities[:, 1]),
                        'accuracy': accuracy_score(y_test, predictions),
                        'recall': recall_score(y_test, predictions, average="micro") if len(
                            classes) > 2 else recall_score(y_test, predictions, average='binary', pos_label=classes[1]),
                        'precision': precision_score(y_test, predictions, average="micro") if len(classes) > 2
                        else precision_score(y_test, predictions, average='binary', pos_label=classes[1]),
                        'f1_score': f1_score(y_test, predictions, average="micro") if len(classes) > 2
                        else f1_score(y_test, predictions, average='binary', pos_label=classes[1]),
                        'execution_time': time.time() - start_time,
                        'tag': 'clean-clean'
                    }
                    logger.info(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
                    self.write_experiment_result_to_db(result)

                logger.info('\n')
