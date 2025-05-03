"""
The `classification_dataset_configs` dictionary contains fundamental information about the CleanML datasets.
The dictionary is structured as follows:
'name': The name of the dataset.
'task': The type of task (e.g., classification, regression).
'target_column': The target variable for the dataset.
'random_seed': A random seed for reproducibility. The random seed is used for train/test splitting.
"""
cleanML_dataset_configs = [
    {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating'},
    # {'name': 'Citation', 'task': 'classification', 'target_column': 'CS'}, # non-informative dataset; only one feature (plain text)
    {'name': 'Company', 'task': 'classification', 'target_column': 'Sentiment'},
    {'name': 'Credit', 'task': 'classification', 'target_column': 'SeriousDlqin2yrs'},
    {'name': 'EEG', 'task': 'classification', 'target_column': 'Eye'},
    {'name': 'Marketing', 'task': 'classification', 'target_column': 'Income'},
    {'name': 'Movie', 'task': 'classification', 'target_column': 'genres'},
    {'name': 'Restaurant', 'task': 'classification', 'target_column': 'priceRange'},
    {'name': 'Sensor', 'task': 'classification', 'target_column': 'moteid'},
    {'name': 'Titanic', 'task': 'classification', 'target_column': 'Survived'},
    {'name': 'University', 'task': 'classification', 'target_column': 'expenses thous$'},
    {'name': 'USCensus', 'task': 'classification', 'target_column': 'Income'},
    {'name': 'KDD', 'task': 'classification', 'target_column': 'is_exciting_20'},
]


from experiment.ExperimentMode import ExperimentMode
from corruption.CorruptionType import CorruptionType

openML_dataset_configs = [
    {
        'name': 'OpenML-CC18',
        'task': 'classification',
        'description': 'OpenML-CC18 benchmark',
        'test_size': 0.3,
        'experiment_modes': [
            ExperimentMode.CLEAN_CLEAN,
            ExperimentMode.CLEAN_DIRTY,
            ExperimentMode.DIRTY_DIRTY
        ],
        'corruptions': [
            CorruptionType.MCAR,
            CorruptionType.SCAR,
            CorruptionType.CSCAR,
        ],
        'row_corruption_percents': [10, 20, 40],
        'column_corruption_percents': [20]
    },
    {
        'name': '8f0ea660163b436bbd4abd49665c7b1d',
        'task': 'regression',
        'description': 'OpenML-CTR23 benchmark',
        'test_size': 0.3,
        'experiment_modes': [
            ExperimentMode.CLEAN_CLEAN,
            ExperimentMode.CLEAN_DIRTY,
            ExperimentMode.DIRTY_DIRTY
        ],
        'corruptions': [
            CorruptionType.MCAR,
            CorruptionType.SCAR,
            CorruptionType.CSCAR,
        ],
        'row_corruption_percents': [10, 20, 40],
        'feature_max_corruption_percent': 20
    },
]
