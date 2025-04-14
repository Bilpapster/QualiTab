from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from tabpfn import TabPFNClassifier
import numpy as np
import time


from load_cleanML_dataset import load_dataset

dataset_config = {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating', "train_test_ratio": 0.5}
print(f"Starting {dataset_config['name']} with {dataset_config['train_test_ratio']} ratio...")

start = time.time()
X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)
print(f"Train-test split finished in {time.time() - start} seconds...")

# Initialize a classifier
clf = TabPFNClassifier()
print(f"Classifier initialized...")
start = time.time()
clf.fit(X_train, y_train)
print(f"Classifier trained in {time.time() - start} seconds...")

SAMPLES = 10
X_train, y_train, X_test, y_test = X_train[:SAMPLES], y_train[:SAMPLES], X_test[:SAMPLES], y_test[:SAMPLES]
print(f"Sample size reduced to {SAMPLES} samples...")

start = time.time()
prediction_probabilities = clf.predict_proba(X_test)
print(f"Prediction probabilities finished in {time.time() - start} seconds...")
print(f"Probabilities shape: {prediction_probabilities.shape}")
print(f"Probabilities type: {type(prediction_probabilities)}")
print()
start = time.time()
predictions = clf.predict(X_test)
print(f"Predictions finished in {time.time() - start} seconds...")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions type: {type(predictions)}")
print()



# Performing inference in batches to avoid out-of-memory issues
MAX_SAMPLES_INFERENCE = 1
prediction_probabilities = []
for i in range(0, len(X_test), MAX_SAMPLES_INFERENCE):
    prediction_probabilities.append(clf.predict_proba(X_test[i:i+MAX_SAMPLES_INFERENCE]))
    print(f"Finished prediction for round {i+1} of test samples")
    print(f"Probabilities shape: {prediction_probabilities.shape}")
    print(f"Probabilities type: {type(prediction_probabilities)}")
    print()

exit()

# Predict probabilities and transform them to predictions (labels) using numpy's argmax function
prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=1)  # there is clf.predict() but is recomputing everything

# print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
print("Accuracy", accuracy_score(y_test, predictions))
print("F1 Score", f1_score(y_test, predictions))
print("Precision", precision_score(y_test, predictions))
print("Recall", recall_score(y_test, predictions))
