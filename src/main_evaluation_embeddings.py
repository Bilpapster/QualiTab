import psycopg2
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
import json
from uuid import uuid4
import openml
from utils import connect_to_db, get_openML_task_mapping

# Table names
EXPERIMENTS_TABLE = "embeddings_experiments"
EVALUATION_TABLE = "embedding_evaluation_metrics"
BATCH_SIZE = 300
K_NEIGHBORS_VALUES = [1, 3, 5, 10]

def calculate_purity(labels_true, labels_pred):
    """Calculates the purity score for clustering."""
    from collections import Counter
    cluster_purity = []
    for cluster_id in np.unique(labels_pred):
        cluster_labels = labels_true[labels_pred == cluster_id]
        most_common = Counter(cluster_labels).most_common(1)
        purity = most_common[0][1] / len(cluster_labels) if len(cluster_labels) > 0 else 0
        cluster_purity.append(purity)
    return np.mean(cluster_purity)


def calculate_knn_similarity(embeddings, k_values):
    """Calculates the average cosine similarity to the k-nearest neighbors."""
    knn_results = {}
    for k in k_values:
        if len(embeddings) < k + 1:
            print(f"Warning: Not enough samples for k={k}. Skipping.")
            continue
        knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')  # +1 to exclude self
        knn.fit(embeddings)
        distances, _ = knn.kneighbors(embeddings)
        # Average distance to the k nearest neighbors (excluding self)
        avg_similarity = np.mean(1 - distances[:, 1:])  # Convert distance to similarity
        knn_results[f'Avg Cosine Similarity (k={k})'] = avg_similarity
    return knn_results


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
    dataset_id = int(dataset_name.split('-')[1])
    random_seed = experiment['random_seed']
    test_embeddings_list = experiment['test_embeddings']
    number_of_classes = experiment['number_of_classes']
    corrupted_rows = experiment['corrupted_rows']
    tag = experiment['tag']

    # if corrupted rows is a nested list, keep only the second element
    if isinstance(corrupted_rows, list) and len(corrupted_rows) > 0 and isinstance(corrupted_rows[0], list):
        print(f"Corrupted rows is a nested list. Unpacking it.")
        corrupted_rows = corrupted_rows[1]

    if not test_embeddings_list:
        print(f"Warning: No test embeddings found for experiment {experiment_id}")
        return []

    test_embeddings = np.array(test_embeddings_list)

    # Retrieve data from OpenML and reproduce transformations and split
    task_to_dataset_id_mapping = get_openML_task_mapping()
    experiment_object = OpenMLExperiment()
    experiment_object.random_seed = random_seed
    experiment_object.task = openml.tasks.get_task(task_to_dataset_id_mapping[dataset_id])
    experiment_object.load_dataset(dataset_config=dict())

    for corrupted_row in corrupted_rows:
        if not corrupted_row in experiment_object.X_test.index.tolist():
            print(f"Warning: Some corrupted rows are not in the test set. The experiment is not reproducible.")
            # write the dataset id to a file
            with open("datasets_to_repeat.txt", "a") as f:
                f.write(f"{dataset_name}\n")
            return []

    # Convert corrupted_rows to a list of locations instead of indices
    corrupted_rows = [experiment_object.X_test.index.get_loc(row) for row in corrupted_rows]

    true_labels = np.array(experiment_object.y_test)

    evaluation_results = []

    if test_embeddings.shape[0] == 2*len(true_labels):
        # keep only the first half of the test embeddings
        test_embeddings = test_embeddings[:len(true_labels)]


    if test_embeddings.shape[0] == len(true_labels) and number_of_classes > 1:
        # Linear Probing
        try:
            logistic_model = LogisticRegression(random_state=random_seed, solver='liblinear', max_iter=1000)
            logistic_model.fit(test_embeddings, experiment_object.y_test)
            y_pred_proba = logistic_model.predict_proba(test_embeddings)
            roc_auc = roc_auc_score(true_labels, y_pred_proba, multi_class='ovr') if number_of_classes > 2 else roc_auc_score(true_labels, y_pred_proba[:, 1])
            evaluation_results.append({
                'evaluation_type': 'linear probing fit to all',
                'metric_name': 'ROC AUC',
                'metric_value': roc_auc,
                'weights': json.dumps(logistic_model.coef_.tolist()) if hasattr(logistic_model, 'coef_') else None
            })
        except ValueError as e:
            print(f"Error during linear probing (all) for experiment {experiment_id}: {e}")

        if corrupted_rows:
            # we have to find the unaffected rows: all the rest values in the test set
            unaffected_rows = [i for i in range(len(test_embeddings)) if i not in corrupted_rows]

            test_embeddings_corrupted = test_embeddings[corrupted_rows]
            y_test_corrupted = experiment_object.y_test.iloc[corrupted_rows]
            test_embeddings_unaffected = test_embeddings[unaffected_rows]
            y_test_unaffected = experiment_object.y_test.iloc[unaffected_rows]
            if len(test_embeddings_corrupted) != len(y_test_corrupted):
                print(f"Warning: The number of corrupted rows does not match the number of test embeddings.")
                print(f"Number of corrupted rows: {len(test_embeddings_corrupted)}")
                print(f"Number of corrupted test embeddings: {len(y_test_corrupted)}")
                exit()
            if len(test_embeddings_unaffected) != len(y_test_unaffected):
                print(f"Warning: The number of unaffected rows does not match the number of test embeddings.")
                print(f"Number of unaffected rows: {len(test_embeddings_unaffected)}")
                print(f"Number of unaffected test embeddings: {len(y_test_unaffected)}")
                exit()

            try:
                # Execute fine-grained linear probing: fit on unaffected, predict on corrupted
                logistic_model = LogisticRegression(random_state=random_seed, solver='liblinear', max_iter=1000)
                logistic_model.fit(test_embeddings_unaffected, y_test_unaffected)
                y_pred_proba = logistic_model.predict_proba(test_embeddings_corrupted)
                roc_auc = roc_auc_score(y_test_corrupted, y_pred_proba, multi_class='ovr') if number_of_classes > 2 else roc_auc_score(y_test_corrupted, y_pred_proba[:, 1])
                evaluation_results.append({
                    'evaluation_type': 'linear probing unaffected-corrupted',
                    'metric_name': 'ROC AUC',
                    'metric_value': roc_auc,
                    'weights': json.dumps(logistic_model.coef_.tolist()) if hasattr(logistic_model, 'coef_') else None
                })
            except ValueError as e:
                print(f"Error during linear probing (unaffected-corrupted) for experiment {experiment_id}: {e}")

            try:
                # Execute fine-grained linear probing: fit on corrupted, predict on unaffected
                logistic_model = LogisticRegression(random_state=random_seed, solver='liblinear', max_iter=1000)
                logistic_model.fit(test_embeddings_corrupted, y_test_corrupted)
                y_pred_proba = logistic_model.predict_proba(test_embeddings_unaffected)
                roc_auc = roc_auc_score(y_test_unaffected, y_pred_proba, multi_class='ovr') if number_of_classes > 2 else roc_auc_score(y_test_unaffected, y_pred_proba[:, 1])
                evaluation_results.append({
                    'evaluation_type': 'linear probing corrupted-unaffected',
                    'metric_name': 'ROC AUC',
                    'metric_value': roc_auc,
                    'weights': json.dumps(logistic_model.coef_.tolist()) if hasattr(logistic_model, 'coef_') else None
                })
            except ValueError as e:
                print(f"Error during linear probing (corrupted-unaffected) for experiment {experiment_id}: {e}")

            try:
                # Execute fine-grained k-means only on unaffected
                kmeans_model = KMeans(n_clusters=number_of_classes, random_state=42, n_init='auto')
                cluster_labels = kmeans_model.fit_predict(test_embeddings_unaffected)
                purity = calculate_purity(y_test_unaffected, cluster_labels)
                evaluation_results.append({
                    'evaluation_type': 'clustering unaffected',
                    'metric_name': 'Purity',
                    'metric_value': purity,
                    'weights': json.dumps(kmeans_model.cluster_centers_.tolist()) if hasattr(kmeans_model,
                                                                                             'cluster_centers_') else None
                })
            except ValueError as e:
                print(f"Error during clustering for experiment {experiment_id}: {e}")

            try:
                # Execute fine-grained k-means only on corrupted
                kmeans_model = KMeans(n_clusters=number_of_classes, random_state=42, n_init='auto')
                cluster_labels = kmeans_model.fit_predict(test_embeddings_corrupted)
                purity = calculate_purity(y_test_corrupted, cluster_labels)
                evaluation_results.append({
                    'evaluation_type': 'clustering corrupted',
                    'metric_name': 'Purity',
                    'metric_value': purity,
                    'weights': json.dumps(kmeans_model.cluster_centers_.tolist()) if hasattr(kmeans_model,
                                                                                             'cluster_centers_') else None
                })
            except ValueError as e:
                print(f"Error during clustering for experiment {experiment_id}: {e}")

            # K-Nearest Neighbors Similarity
            knn_results = calculate_knn_similarity(test_embeddings_unaffected, K_NEIGHBORS_VALUES)
            for metric_name, metric_value in knn_results.items():
                evaluation_results.append({
                    'evaluation_type': 'knn similarity',
                    'metric_name': metric_name + " (unaffected)",
                    'metric_value': metric_value,
                    'weights': None
                })

            # K-Nearest Neighbors Similarity
            knn_results = calculate_knn_similarity(test_embeddings_corrupted, K_NEIGHBORS_VALUES)
            for metric_name, metric_value in knn_results.items():
                evaluation_results.append({
                    'evaluation_type': 'knn similarity',
                    'metric_name': metric_name + " (corrupted)",
                    'metric_value': metric_value,
                    'weights': None
                })



        # Clustering (K-Means)
        try:
            # Execute k-means clustering on the entire test set
            kmeans_model = KMeans(n_clusters=number_of_classes, random_state=42, n_init='auto')
            cluster_labels = kmeans_model.fit_predict(test_embeddings)
            purity = calculate_purity(true_labels, cluster_labels)
            evaluation_results.append({
                'evaluation_type': 'clustering all test',
                'metric_name': 'Purity',
                'metric_value': purity,
                'weights': json.dumps(kmeans_model.cluster_centers_.tolist()) if hasattr(kmeans_model, 'cluster_centers_') else None
            })
        except ValueError as e:
            print(f"Error during clustering for experiment {experiment_id}: {e}")

        # K-Nearest Neighbors Similarity
        knn_results = calculate_knn_similarity(test_embeddings, K_NEIGHBORS_VALUES)
        for metric_name, metric_value in knn_results.items():
            evaluation_results.append({
                'evaluation_type': 'knn similarity',
                'metric_name': metric_name + " (all)",
                'metric_value': metric_value,
                'weights': None
            })

    else:
        print(f"Warning: Inconsistent data or single class for experiment {experiment_id}, skipping evaluation.")
        print(f"------------- Shape of test embeddings: {test_embeddings.shape} -------------")
        print(f"------------- Shape of true labels: {len(true_labels)} -------------")
        print(f"------------- Number of classes: {number_of_classes} -------------")
        print(f"------------- Experiment ID: {experiment_id} -------------")
        print(f"------------- Shape of X_train: {experiment_object.X_train.shape} -------------")
        print(f"------------- Shape of y_train: {experiment_object.y_train.shape} -------------")
        print(f"------------- Shape of X_test: {experiment_object.X_test.shape} -------------")
        print(f"------------- Shape of y_test: {experiment_object.y_test.shape} -------------")
        print()

    return evaluation_results

def store_evaluation_metrics(conn, result, experiment_id):
    """Stores a single evaluation metric in the embedding_evaluation_metrics table."""
    cursor = conn.cursor()
    evaluation_id = uuid4()
    sql = f"""
        INSERT INTO {EVALUATION_TABLE} (evaluation_id, experiment_id, evaluation_type, metric_name, metric_value, weights)
        VALUES (%s, %s, %s, %s, %s, %s);
    """
    cursor.execute(sql, (str(evaluation_id), str(experiment_id), result['evaluation_type'], result['metric_name'], result['metric_value'], result.get('weights')))
    conn.commit()

def main():
    try:
        conn, cursor = connect_to_db()
        offset = 0
        processed_count = 0
        all_experiment_ids = set()

        # Get all existing experiment_ids in the evaluation table to avoid duplicates
        cursor.execute(f"SELECT DISTINCT experiment_id FROM {EVALUATION_TABLE}")
        existing_evaluated_ids = {row[0] for row in cursor.fetchall()}

        while True:
            cursor.execute(f"""
            SELECT experiment_id, dataset_name, random_seed, test_embeddings, number_of_classes, corrupted_rows, tag
            FROM {EXPERIMENTS_TABLE} 
            ORDER BY experiment_id DESC
            OFFSET %s LIMIT %s""", (offset, BATCH_SIZE))
            batch = [dict(zip([column.name for column in cursor.description], row)) for row in cursor.fetchall()]

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

            offset += BATCH_SIZE

        cursor.close()
        conn.close()
        print(f"Evaluation process completed. Processed {processed_count} new experiments.")

    except psycopg2.Error as e:
        print(f"Error connecting to or querying the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
