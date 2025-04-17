import sys
sys.path.append("..")

from .Experiment import Experiment
from src.utils import configure_logging

import time, uuid
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


class ClassificationExperiment(Experiment):
    def insertion_query(self, result: dict) -> dict:
        return {
            'query': """
                     INSERT INTO classification_experiments (experiment_id,
                                                             dataset_name,
                                                             train_size,
                                                             test_size,
                                                             used_default_split,
                                                             random_seed,
                                                             roc_auc,
                                                             accuracy,
                                                             recall,
                                                             precision,
                                                             f1_score,
                                                             execution_time,
                                                             tag)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                     """,
            'vars': (
                str(uuid.uuid4()),
                result['dataset'],
                result['train_size'],
                result['test_size'],
                result['used_default_split'],
                result['random_seed'],
                result['roc_auc'],
                result['accuracy'],
                result['recall'],
                result['precision'],
                result['f1_score'],
                result['execution_time'],
                "dirty-dirty",
            ),
        }

    def run(self):
        """
        Main method to run the experiment.
        """
        MAX_SAMPLES_INFERENCE = int(1e4)  # TabPFN's limit is 10K data samples
        dataset_configs = self.get_dataset_configs()
        logger = configure_logging()

        for dataset_config in dataset_configs:
            for random_seed in self.get_random_seeds():
                start_time = time.time()
                logger.info(f"Working on dataset {dataset_config['name']} with random seed {random_seed}.")

                X_train, y_train, X_test, y_test, used_default_split, random_seed = self.load_dataset(dataset_config,
                                                                                                      random_seed=random_seed)
                logger.info(f"Data split to train and test set.")

                clf = self.get_model_from_dataset_config(dataset_config)
                clf.fit(X_train, y_train)
                classes = clf.classes_
                logger.info(f"TabPFN classifier fitted. The target classes are {classes}")
                # Perform inference in batches to avoid out-of-memory issues
                prediction_probabilities = None
                logger.info(f"Starting inference on {len(X_test)} samples in batches of {MAX_SAMPLES_INFERENCE}.")
                for i in range(0, len(X_test), MAX_SAMPLES_INFERENCE):
                    batch = X_test[i:i + MAX_SAMPLES_INFERENCE]
                    # batch_probabilities = clf.predict_proba(batch)
                    batch_probabilities = np.random.randn(len(batch), 2)

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
                    'dataset': dataset_config['name'],
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'used_default_split': used_default_split,
                    'random_seed': random_seed,
                    # Caution: ROC AUC is valid this way only for binary classification!
                    'roc_auc': roc_auc_score(y_test, prediction_probabilities[:, 1]),
                    'accuracy': accuracy_score(y_test, predictions),
                    'recall': recall_score(y_test, predictions, average="binary", pos_label=classes[1]),
                    'precision': precision_score(y_test, predictions, average="binary", pos_label=classes[1]),
                    'f1_score': f1_score(y_test, predictions, average="binary", pos_label=classes[1]),
                    'execution_time': time.time() - start_time,
                }
                logger.info(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
                self.write_experiment_result_to_db(result)
