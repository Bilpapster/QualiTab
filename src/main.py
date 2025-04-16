import psycopg2  # For database interaction
import json # For storing embeddings
import uuid
import os
from load_cleanML_dataset import load_dataset, extract_embeddings
from load_cleanML_dataset_improved import scan_for_clean_data
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

def run_experiments2(dataset_configs):
    for dataset in dataset_configs:
        sources = scan_for_clean_data(dataset.get('name'))
        print(type(sources))
        for source in sources:
            print(len(source)) # source contains 4 things: train, test, path, random seed
            # print(type(source[2]))
            print(source[2])
        exit()
        # print(type(sources))
        # print(f"Datasets found: {sources}")
        print()


if __name__ == "__main__":
    from cleanML_dataset_configs import classification_dataset_configs
    run_experiments2(classification_dataset_configs)