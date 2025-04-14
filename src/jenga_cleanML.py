from jenga.basis import BinaryClassificationTask
from typing import Any, Dict, List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    make_scorer,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from load_cleanML_dataset import load_dataset

dataset_config = {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating', "train_test_ratio": 0.5}
X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)
class_map = {'N': 0, 'Y': 1}
y_train.replace(class_map, inplace=True)
y_test.replace(class_map, inplace=True)
y_train, y_test = y_train.astype('category'), y_test.astype('category')
categorical_features = ['LocationName']
numerical_features = list(X_train).copy()
for categorical_feature in categorical_features:
    numerical_features.remove(categorical_feature)

task = BinaryClassificationTask(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    categorical_columns=categorical_features,
    numerical_columns=numerical_features,
    seed=42
)

# Source code from https://github.com/schelterlabs/jenga/blob/master/src/jenga/basis.py#L229
# Important: scikit-learn's version must be exactly 1.3.1.
# Needed to selectively copy and adjust code from Jenga, because they were using 'log' loss which is invalid.
# Replaced it with 'log_loss' below and this solves the problem.

categorical_preprocessing = Pipeline(
    [
        ('mark_missing', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
    ]
)

numerical_preprocessing = Pipeline(
    [
        ('mark_missing', SimpleImputer(strategy='mean')),
        ('scaling', StandardScaler())
    ]
)

feature_transformation = ColumnTransformer(transformers=[
    ('categorical_features', categorical_preprocessing, categorical_features),
    ('scaled_numeric', numerical_preprocessing, numerical_features),
])


def _get_pipeline_grid_scorer_tuple(
        feature_transformation: ColumnTransformer
) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
    """
    Binary classification specific default `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.

    Args:
        feature_transformation (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

    Returns:
        Tuple[Dict[str, object], Any, Dict[str, Any]]: Binary classification specific parts to build baseline model
    """

    param_grid = {
        # Jenga here uses 'log' which is invalid and the program crashes. 'log_loss' solves the problem. 'modified_huber' also works.
        'learner__loss': ['log_loss', 'modified_huber'],
        'learner__penalty': ['l2'],
        # Jenga's original: [0.00001, 0.0001, 0.001, 0.01]
        'learner__alpha': [
            0.00001, 0.00003,
            0.0001, 0.0003,
            0.001, 0.003,
            0.01, 0.03
        ]
    }

    pipeline = Pipeline(
        [
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000, n_jobs=-1))
        ]
    )

    scorer = {
        "ROC/AUC": make_scorer(roc_auc_score, needs_proba=True)
    }

    return param_grid, pipeline, scorer


param_grid, pipeline, scorer = _get_pipeline_grid_scorer_tuple(feature_transformation)
refit = list(scorer.keys())[0]

search = GridSearchCV(pipeline, param_grid, scoring=scorer, n_jobs=-1, refit=refit)
model = search.fit(X_train, y_train).best_estimator_

predicted_label_probabilities = model.predict_proba(X_test)
score = roc_auc_score(y_test, predicted_label_probabilities[:, 1])

print(f"The ROC AUC score on the test data is {score}")
