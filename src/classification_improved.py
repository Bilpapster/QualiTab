import time
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from load_cleanML_dataset import load_dataset
from cleanML_dataset_configs import classification_dataset_configs
from utils import configure_logging

logger = configure_logging()
MAX_SAMPLES_INFERENCE = int(1e4) # TabPFN's limit is 10K data samples

for dataset_config in classification_dataset_configs:
    start_time = time.time()
    logger.info(f"Working on dataset {dataset_config['name']}.")

    X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)
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
        batch_probabilities = clf.predict_proba(batch)
        # batch_probabilities = np.random.randn(len(batch), 2)

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
    print(result)
    print()

