import psycopg2  # For database interaction
import json # For storing embeddings
import uuid
import os
from load_cleanML_dataset import load_dataset, extract_embeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_random_seeds_list(s):
    """
    Converts a comma-separated string of integers into a list of ints.
    Example: "100,200,300..." -> [100, 200, 300, ...]
    """
    return [int(x.strip()) for x in s.strip().split(',')]

def run_experiment(dataset_configs):
    """Runs experiments and stores results in the database."""
    conn = psycopg2.connect(
        host=os.getenv("POSTRGES_HOST", "localhost"),
        user=os.getenv("POSTRGES_USER", "postgres"),
        password=os.getenv("POSTRGES_PASSWORD", "postgres"),
        database=os.getenv("POSTRGES_DB", "postgres"),
    )
    cursor = conn.cursor()

    for dataset_config in dataset_configs:
        try:
            X_train, y_train, X_test, y_test, used_default_split, random_seed = load_dataset(dataset_config)
            embeddings, start_time, end_time = extract_embeddings(X_train, y_train, X_test, dataset_config['task'])
            print(embeddings)

            experiment_id = uuid.uuid4() # Generate a unique ID for the experiment
            print(f"Experiment ID: {experiment_id}")
            print(f"Dataset config: {dataset_config}")
            print(f"Embeddings: {embeddings}")
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            print()

            # Insert experiment data into the database
            cursor.execute(
                "INSERT INTO experiments (experiment_id, dataset_name, tag, start_time, end_time, n_train, n_test, used_default_split, random_seed, embeddings) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (str(experiment_id), dataset_config['name'], "dirty-dirty", int(start_time), int(end_time), len(X_train), len(X_test), used_default_split, random_seed, json.dumps(embeddings.tolist())), # Convert embeddings to JSON for storage
            )
            conn.commit()
            logger.info(f"Experiment for {dataset_config['name']} completed and saved to database.")

        except Exception as e:
            # conn.rollback()  # Rollback in case of error
            logger.error(f"Error processing {dataset_config['name']}: {e}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    dataset_configs = [
        {'name': 'Airbnb', 'task': 'classification', 'target_column': 'Rating'},
        {'name': 'Citation', 'task': 'classification', 'target_column': 'CS'},
        {'name': 'Company', 'task': 'classification', 'target_column': 'Sentiment'},
        {'name': 'Credit', 'task': 'classification', 'target_column': 'SeriousDlqin2yrs'},
        {'name': 'EEG', 'task': 'classification', 'target_column': 'Eye'},
        # {'name': 'KDD', 'task': 'classification', 'target_column': 'is_exciting_20'}, # for some reason runs out of memory, so temporarily is commented out
        {'name': 'Marketing', 'task': 'classification', 'target_column': 'Income'},
        {'name': 'Movie', 'task': 'classification', 'target_column': 'genres'},
        {'name': 'Restaurant', 'task': 'classification', 'target_column': 'priceRange'},
        {'name': 'Sensor', 'task': 'classification', 'target_column': 'moteid'},
        {'name': 'Titanic', 'task': 'classification', 'target_column': 'Survived'},
        {'name': 'University', 'task': 'classification', 'target_column': 'expenses thous$'},
        {'name': 'USCensus', 'task': 'classification', 'target_column': 'Income'},
    ]
    run_experiment(dataset_configs)