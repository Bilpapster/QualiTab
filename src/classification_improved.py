import os, time, uuid
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from load_cleanML_dataset import load_dataset
from cleanML_dataset_configs import classification_dataset_configs
from utils import configure_logging, connect_to_db, get_seeds_from_env_or_else_default

logger = configure_logging()


def classification_insertion_query(result: dict) -> dict:
   """
    Returns the SQL query for inserting classification experiment results into the database
    in the form of a dictionary that contains the query and the values to be inserted.
   """
   return {
       'query': """
            INSERT INTO classification_experiments (
                experiment_id,
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
                tag
            )
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


def write_classification_experiment_result_to_db(result: dict):
    """
    Writes the classification experiment result to the database.
    It connects to the PostgreSQL database using environment variables,
    inserts the experiment data into the classification_experiments table,
    and logs the success or failure of the operation.
    """
    conn, cursor = connect_to_db()
    try:
        cursor.execute(**classification_insertion_query(result))
        conn.commit()
        logger.info(f"Experiment for dataset {result['dataset']} saved to database.")
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        logger.error(f"Error processing this result configuration: {result}: {e}")


def run_classification_experiments_on_CleanML():
    MAX_SAMPLES_INFERENCE = int(1e4)  # TabPFN's limit is 10K data samples

    for dataset_config in classification_dataset_configs:
        for random_seed in get_seeds_from_env_or_else_default():
            start_time = time.time()
            logger.info(f"Working on dataset {dataset_config['name']} with random seed {random_seed}.")

            X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config, mode='force_manual_split')
            logger.info(f"Data split to train and test set.")

            # Initialize a classifier
            clf = TabPFNClassifier()
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
                    prediction_probabilities = np.concatenate((prediction_probabilities, batch_probabilities), axis=0)
                logger.info(f"Finished inference for samples {i} -- {i + len(batch) - 1}")
            logger.info(f"Inference finished.")
            predictions = np.array(classes)[np.argmax(prediction_probabilities, axis=1)]
            logger.info(f"Prediction probabilities successfully converted to predictions of shape {predictions.shape}.")

            # Evaluate the model and return the results as a dictionary
            result = {
                'dataset': dataset_config['name'],
                'train_size': len(X_train),
                'test_size': len(X_test),
                'used_default_split': used_default_split,
                'random_seed': random_seed,
                'roc_auc': roc_auc_score(y_test, prediction_probabilities[:, 1]), # Caution: only for binary classification!
                'accuracy': accuracy_score(y_test, predictions),
                'recall': recall_score(y_test, predictions, average="binary", pos_label=classes[1]),
                'precision': precision_score(y_test, predictions, average="binary", pos_label=classes[1]),
                'f1_score': f1_score(y_test, predictions, average="binary", pos_label=classes[1]),
                'execution_time': time.time() - start_time,
            }
            logger.info(f"Dataset finished. Execution time: {result['execution_time']} seconds.")
            write_classification_experiment_result_to_db(result)


if __name__ == "__main__":
    run_classification_experiments_on_CleanML()
