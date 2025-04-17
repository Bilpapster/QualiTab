"""
The `classification_dataset_configs` dictionary contains fundamental information about the CleanML datasets.
The dictionary is structured as follows:
'name': The name of the dataset.
'task': The type of task (e.g., classification, regression).
'target_column': The target variable for the dataset.
'random_seed': A random seed for reproducibility. The random seed is used for train/test splitting.
"""

classification_dataset_configs = [
    {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating', 'random_seed':260507},
    # {'name': 'Citation', 'task': 'classification', 'target_column': 'CS', 'random_seed':260507}, # non-informative dataset; only one feature (plain text)
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
