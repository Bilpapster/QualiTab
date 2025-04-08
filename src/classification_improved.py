from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from tabpfn import TabPFNClassifier
import numpy as np

from load_cleanML_dataset import load_dataset

dataset_config = {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating', "train_test_ratio": 0.5}
X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)


# Performing inference in batches to avoid out-of-memory issues
MAX_SAMPLES_INFERENCE = 100
prediction_probabilities = []
for i in range(0, len(X_test), MAX_SAMPLES_INFERENCE):
    prediction_probabilities.append(clf.predict_proba(X_test[i:i+MAX_SAMPLES_INFERENCE]))
    print(f"Finished prediction for round {i} of test samples")


# Predict probabilities and transform them to predictions (labels) using numpy's argmax function
# prediction_probabilities = clf.predict_proba(X_test)
predictions = np.argmax(prediction_probabilities, axis=1)  # there is clf.predict() but is recomputing everything

print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
print("Accuracy", accuracy_score(y_test, predictions))
print("F1 Score", f1_score(y_test, predictions))
print("Precision", precision_score(y_test, predictions))
print("Recall", recall_score(y_test, predictions))
