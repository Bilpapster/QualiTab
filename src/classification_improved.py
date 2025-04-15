import time, logging, numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from load_cleanML_dataset import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(filename)s::%(funcName)s::%(lineno)d %(asctime)s - %(levelname)s - %(message)s - ')
logger = logging.getLogger(__name__)


dataset_configs = [
    {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating', 'random_seed':260507},
    {'name': 'Citation', 'task': 'classification', 'target_column': 'CS', 'random_seed':260507},
    {'name': 'Company', 'task': 'classification', 'target_column': 'Sentiment', 'random_seed':260507},
    {'name': 'Credit', 'task': 'classification', 'target_column': 'SeriousDlqin2yrs', 'random_seed':260507},
    {'name': 'EEG', 'task': 'classification', 'target_column': 'Eye', 'random_seed':260507},
    {'name': 'Marketing', 'task': 'classification', 'target_column': 'Income', 'random_seed':260507},
    {'name': 'Movie', 'task': 'classification', 'target_column': 'genres', 'random_seed':260507},
    {'name': 'Restaurant', 'task': 'classification', 'target_column': 'priceRange', 'random_seed':260507},
    {'name': 'Sensor', 'task': 'classification', 'target_column': 'moteid', 'random_seed':260507},
    {'name': 'Titanic', 'task': 'classification', 'target_column': 'Survived', 'random_seed':260507},
    {'name': 'University', 'task': 'classification', 'target_column': 'expenses thous$', 'random_seed':260507},
    {'name': 'USCensus', 'task': 'classification', 'target_column': 'Income', 'random_seed':260507},
    {'name': 'KDD', 'task': 'classification', 'target_column': 'is_exciting_20', 'random_seed': 260507},
]
for dataset_config in dataset_configs:
    start_time = time.time()
    logger.info(f"Working on dataset {dataset_config['name']}.")

    X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)
    logger.info(f"Data split to train and test set.")

    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    classes = clf.classes_
    logger.info(f"TabPFN classifier fitted. The target classes are {classes}")

    # Performing inference in batches to avoid out-of-memory issues
    MAX_SAMPLES_INFERENCE = int(1e4)
    prediction_probabilities = None
    logger.info(f"Starting inference on {len(X_test)} samples in batches of {MAX_SAMPLES_INFERENCE}.")
    for i in range(0, len(X_test), MAX_SAMPLES_INFERENCE):
        batch = X_test[i:i + MAX_SAMPLES_INFERENCE]
        batch_probabilities = clf.predict_proba(batch)

        if prediction_probabilities is None:
            prediction_probabilities = np.array(batch_probabilities)
            continue
        prediction_probabilities = np.concatenate((prediction_probabilities, batch_probabilities), axis=0)
        logger.info(f"\tFinished inference for samples {i} -- {i + len(X_train) - 1}")
    logger.info(f"Inference finished.")
    predictions = np.argmax(prediction_probabilities, axis=1)
    predictions = np.array([classes[i] for i in predictions])
    logger.info(f"Prediction probabilities are successfully converted to predictions of shape {predictions.shape}.")
    # print(predictions)
    print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
    print("Accuracy", accuracy_score(y_test, predictions))
    print(f"Total time: {time.time() - start_time}")
    print()
    # print("F1 Score", f1_score(y_test, predictions))
    # print("Precision", precision_score(y_test, predictions))
    # print("Recall", recall_score(y_test, predictions))
