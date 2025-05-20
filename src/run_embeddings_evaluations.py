import psycopg2
import numpy as np
from uuid import uuid4
import openml

from utils import (
    connect_to_db,
    get_idx_positions_from_idx_values,
    keep_corrupted_rows_only_for_test_set,
    fetch_corrupted_rows_for_clean_clean_experiment,
    fetch_all_as_list_of_dicts,
    fetch_clean_clean_embeddings,
)
from config import naming_config as names
from evaluation import (
    linear_probing,
    euclidean_distances_from_reference,
    cosine_distance_from_reference,
    roc_auc as auc
)

# Table names
EXPERIMENTS_TABLE = "embeddings_experiments"
EVALUATION_TABLE = "embedding_evaluation_metrics"
BATCH_SIZE = 300
K_NEIGHBORS_VALUES = [1, 3, 5, 10]


def evaluate_embeddings(experiment):
    """
    Evaluates the embeddings from a single experiment using linear probing, clustering, and KNN.

    Args:
        experiment (dict): A dictionary representing a row from the embeddings_experiments table.

    Returns:
        list: A list of dictionaries containing the evaluation metrics.
    """
    from experiment.OpenMLExperiment import OpenMLExperiment

    experiment_id = experiment['experiment_id']
    dataset_name = str(experiment['dataset_name'])
    task_id = OpenMLExperiment.get_task_id_from_dataset_name(dataset_name)
    random_seed = experiment['random_seed']
    test_embeddings_list = experiment['test_embeddings']
    number_of_classes = experiment['number_of_classes']
    error_type = experiment['error_type']
    row_corruption_percent = experiment['row_corruption_percent']
    column_corruption_percent = experiment['column_corruption_percent']
    tag = experiment['tag']
    corrupted_rows = experiment['corrupted_rows'] \

    row_corruption_percents = [row_corruption_percent] if tag != 'CLEAN_CLEAN' else [0, 10, 20, 40]

    evaluation_results = []
    for row_corruption_percent in row_corruption_percents:
        if tag == 'CLEAN_CLEAN' and row_corruption_percent > 0:
            corrupted_rows = fetch_corrupted_rows_for_clean_clean_experiment(
                dataset_name=dataset_name,
                random_seed=random_seed,
                row_corruption_percent=row_corruption_percent,
                column_corruption_percent=column_corruption_percent,
            )
            column_corruption_percent = 20

        corrupted_rows = keep_corrupted_rows_only_for_test_set(corrupted_rows)

        if not test_embeddings_list:
            print(f"Warning: No test embeddings found for experiment {experiment_id}")
            return []

        test_embeddings = np.array(test_embeddings_list)

        # Retrieve data from OpenML and reproduce transformations and split
        experiment_object = OpenMLExperiment()
        experiment_object.random_seed = random_seed
        experiment_object.task = openml.tasks.get_task(task_id)
        experiment_object.load_dataset(dataset_config=dict())

        # Convert corrupted_rows to a list of locations (positions) instead of index values
        corrupted_rows = get_idx_positions_from_idx_values(corrupted_rows, experiment_object.X_test)
        clean_rows = [i for i in range(len(experiment_object.X_test)) if i not in corrupted_rows]
        true_labels = np.array(experiment_object.y_test)
        clean_corrupted_labels = np.array([1 if i in corrupted_rows else 0 for i in range(len(test_embeddings))])


        if test_embeddings.shape[0] > len(true_labels):
            # keep only the first half of the test embeddings
            print(f"Warning: The number of test embeddings is greater than the number of true labels for experiment {experiment_id}. Keeping only the first embeddings to match the number.")
            test_embeddings = test_embeddings[:len(true_labels)]


        if test_embeddings.shape[0] < len(true_labels) or number_of_classes <= 1:
            # If the number of test embeddings is not equal to the number of true labels, or if there's only one class,
            # we cannot proceed with linear probing or clustering.
            print(f"Warning: Inconsistent data for experiment {experiment_id}, skipping evaluation.")
            print(f"Number of test embeddings: {len(test_embeddings)}")
            print(f"Number of true labels: {len(true_labels)}")
            print(f"Number of classes: {number_of_classes}")
            return []

        try:
            # linear probing on all test set
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.LINEAR_PROBING,
                    samples_type=names.SamplesType.ALL,
                    ground_truth_assignment=names.GroundTruthAssignment.TASK_LABELS,
                ),
                'metric_name': names.EvaluationMetric.ROC_AUC.value,
                'metric_value': auc(
                    true_labels,
                    linear_probing(test_embeddings, true_labels, random_state=random_seed)
                ),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Warning during linear probing on all test set for experiment {experiment_id}: {e}")
            pass

        try:
            # linear probing on corrupted only
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.LINEAR_PROBING,
                    samples_type=names.SamplesType.CORRUPTED,
                    ground_truth_assignment=names.GroundTruthAssignment.TASK_LABELS,
                ),
                'metric_name': names.EvaluationMetric.ROC_AUC.value,
                'metric_value': auc(
                    true_labels[corrupted_rows],
                    linear_probing(test_embeddings[corrupted_rows], true_labels[corrupted_rows], random_state=random_seed)
                ),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during linear probing on corrupted for experiment {experiment_id}: {e}")
            pass


        try:
            # linear probing on clean only
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.LINEAR_PROBING,
                    samples_type=names.SamplesType.CLEAN,
                    ground_truth_assignment=names.GroundTruthAssignment.TASK_LABELS,
                ),
                'metric_name': names.EvaluationMetric.ROC_AUC.value,
                'metric_value': auc(
                    true_labels[clean_rows],
                    linear_probing(test_embeddings[clean_rows], true_labels[clean_rows], random_state=random_seed)
                ),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during linear probing on clean for experiment {experiment_id}: {e}")
            pass

        try:
            # linear probing to classify clean from corrupted
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.LINEAR_PROBING,
                    samples_type=names.SamplesType.ALL,
                    ground_truth_assignment=names.GroundTruthAssignment.CLEAN_CORRUPTED,
                ),
                'metric_name': names.EvaluationMetric.ROC_AUC.value,
                'metric_value': auc(
                    clean_corrupted_labels,
                    linear_probing(test_embeddings, clean_corrupted_labels, random_state=random_seed)
                ),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during linear probing to classify clean from corrupted for experiment {experiment_id}: {e}")
            pass

        if tag == 'CLEAN_CLEAN':
            # No other evaluation is meaningful for clean-clean experiments
            continue

        clean_clean_embeddings = np.array(fetch_clean_clean_embeddings(dataset_name=dataset_name, random_seed=random_seed))
        euclidean_distances = np.array(euclidean_distances_from_reference(clean_clean_embeddings, test_embeddings, z_normalize_first=True))
        cosine_similarities = np.array(cosine_distance_from_reference(clean_clean_embeddings, test_embeddings))

        try:
            # Avg Z-norm Euclidean distance from respective embeddings when they are clean (reference) for all samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.ALL,
                ),
                'metric_name': names.EvaluationMetric.Z_NORM_EUCLIDEAN_DISTANCE.value,
                'metric_value': np.mean(euclidean_distances),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during Avg Z-norm Euclidean distance from respective embeddings when they are clean (reference) for all samples for experiment {experiment_id}: {e}")
            pass

        try:
            # Avg Z-norm Euclidean distance from respective embeddings when they are clean (reference) for corrupted samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.CORRUPTED,
                ),
                'metric_name': names.EvaluationMetric.Z_NORM_EUCLIDEAN_DISTANCE.value,
                'metric_value': np.mean(euclidean_distances[corrupted_rows]),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during evaluating embeddings for experiment {experiment_id}: {e}")
            pass

        try:
            # Avg Z-norm Euclidean distance from respective embeddings when they are clean (reference) for clean samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.CLEAN,
                ),
                'metric_name': names.EvaluationMetric.Z_NORM_EUCLIDEAN_DISTANCE.value,
                'metric_value': np.mean(euclidean_distances[clean_rows]),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during Avg Z-norm Euclidean distance from respective embeddings when they are clean (reference) for clean samples for experiment {experiment_id}: {e}")
            pass

        try:
            # Avg Cosine Similarity from respective embeddings when they are clean (reference) for all samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.ALL,
                ),
                'metric_name': names.EvaluationMetric.COSINE_SIMILARITY.value,
                'metric_value': np.mean(cosine_similarities),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during Avg Cosine Similarity from respective embeddings when they are clean (reference) for all samples for experiment {experiment_id}: {e}")
            pass

        try:
            # Avg Cosine Similarity from respective embeddings when they are clean (reference) for corrupted samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.CORRUPTED,
                ),
                'metric_name': names.EvaluationMetric.COSINE_SIMILARITY.value,
                'metric_value': np.mean(cosine_similarities[corrupted_rows]),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during Avg Cosine Similarity from respective embeddings when they are clean (reference) for corrupted samples for experiment {experiment_id}: {e}")
            pass

        try:
            # Avg Cosine Similarity from respective embeddings when they are clean (reference) for clean samples
            evaluation_results.append({
                'evaluation_type': names.get_standardized_name_from_configurations(
                    evaluation_task=names.EvaluationTask.DISTANCE_FROM_REFERENCE,
                    samples_type=names.SamplesType.CLEAN,
                ),
                'metric_name': names.EvaluationMetric.COSINE_SIMILARITY.value,
                'metric_value': np.mean(cosine_similarities[clean_rows]),
                'row_corruption_percent': row_corruption_percent,
            })
        except Exception as e:
            print(f"Error during Avg Cosine Similarity from respective embeddings when they are clean (reference) for clean samples for experiment {experiment_id}: {e}")
            pass

    return evaluation_results

def store_evaluation_metrics(conn, result, experiment_id):
    """Stores a single evaluation metric in the embedding_evaluation_metrics table."""
    cursor = conn.cursor()
    evaluation_id = uuid4()
    sql = f"""
        INSERT INTO {EVALUATION_TABLE} (evaluation_id, experiment_id, evaluation_type, metric_name, metric_value, row_corruption_percent)
        VALUES (%s, %s, %s, %s, %s, %s);
    """
    cursor.execute(sql, (str(evaluation_id), str(experiment_id), result['evaluation_type'], result['metric_name'], result['metric_value'], result.get('row_corruption_percent', None)))
    conn.commit()

def main():
    try:
        conn, cursor = connect_to_db()
        cursor.execute('''SELECT DISTINCT experiment_id FROM embeddings_experiments;''')

        experiment_ids = set([id[0] for id in cursor.fetchall()])
        all_experiment_ids = set()
        processed_count = 0

        # Get all existing experiment_ids in the evaluation table to avoid duplicates
        cursor.execute(f"SELECT DISTINCT experiment_id FROM {EVALUATION_TABLE}")
        existing_evaluated_ids = {row[0] for row in cursor.fetchall()}

        for experiment_id in experiment_ids:
        # while True:
            cursor.execute(f"""
            SELECT experiment_id, dataset_name, random_seed, test_embeddings, number_of_classes, corrupted_rows, tag,
            error_type, row_corruption_percent, column_corruption_percent
            FROM {EXPERIMENTS_TABLE} 
            WHERE experiment_id = '{experiment_id}';""")

            batch = fetch_all_as_list_of_dicts(cursor)

            if not batch:
                break

            for experiment in batch:
                experiment_id = experiment['experiment_id']
                if experiment_id not in existing_evaluated_ids and experiment_id not in all_experiment_ids:
                    results = evaluate_embeddings(experiment)
                    if results:
                        for result in results:
                            store_evaluation_metrics(conn, result, experiment_id)
                        all_experiment_ids.add(experiment_id)
                        print(f"Evaluated experiment {experiment_id}")
                    processed_count += 1
                else:
                    print(f"Skipping already evaluated experiment {experiment_id}")

        cursor.close()
        conn.close()
        print(f"Evaluation process completed. Processed {processed_count} new experiments.")

    except psycopg2.Error as e:
        print(f"Error connecting to or querying the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
