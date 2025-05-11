import psycopg2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
import json
from uuid import uuid4
import openml
from utils import connect_to_db, get_openML_task_mapping

# Table names
EXPERIMENTS_TABLE = "embeddings_experiments"
EVALUATION_TABLE = "embedding_evaluation_metrics"

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

def evaluate_embeddings(experiment):
    """
    Evaluates the embeddings from a single experiment using linear probing and clustering.

    Args:
        experiment (dict): A dictionary representing a row from the embeddings_experiments table.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    from experiment.OpenMLExperiment import OpenMLExperiment

    experiment_id = experiment['experiment_id']
    dataset_name = str(experiment['dataset_name'])
    dataset_id = int(dataset_name.split('-')[1])
    random_seed = experiment['random_seed']
    test_embeddings_list = experiment['test_embeddings']
    number_of_classes = experiment['number_of_classes']

    print(type(test_embeddings_list))

    if not test_embeddings_list:
        print(f"Warning: No test embeddings found for experiment {experiment_id}")
        return []

    # test_embeddings_list = json.loads(test_embeddings_json)
    test_embeddings = np.array([np.array(emb) for emb in test_embeddings_list])
    # exit()

    # Assuming you have a way to retrieve the true labels for the test set
    # This might involve querying another table or storing them in the experiments table
    # For now, let's assume 'true_labels' is available in the experiment dictionary
    task_to_dataset_id_mapping = get_openML_task_mapping()
    experiment_object = OpenMLExperiment()
    experiment_object.random_seed = random_seed
    experiment_object.task = openml.tasks.get_task(task_to_dataset_id_mapping[dataset_id])
    experiment_object.load_dataset(dataset_config=dict()) # uses default train test split (70/30)

    true_labels = np.array(experiment_object.y_test)

    evaluation_results = []

    if test_embeddings.shape[0] == len(true_labels) and number_of_classes > 1:
        # Linear Probing (Logistic Regression)
        try:
            # X_train, X_test, y_train, y_test = train_test_split(test_embeddings, true_labels, test_size=0.2, random_state=42)
            logistic_model = LogisticRegression(random_state=random_seed, solver='liblinear', multi_class='ovr', max_iter=1000)
            # logistic_model.fit(X_train, y_train)
            # y_pred_proba = logistic_model.predict_proba(X_test)
            logistic_model.fit(test_embeddings, experiment_object.y_test)
            y_pred_proba = logistic_model.predict_proba(test_embeddings)

            # Handle binary vs. multi-class
            if number_of_classes == 2:
                # roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                roc_auc = roc_auc_score(experiment_object.y_test, y_pred_proba[:, 1])
            else:
                # roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                roc_auc = roc_auc_score(experiment_object.y_test, y_pred_proba, multi_class='ovr')

            evaluation_results.append({
                'evaluation_type': 'linear probing',
                'metric_name': 'ROC AUC',
                'metric_value': roc_auc,
                'weights': json.dumps(logistic_model.coef_.tolist()) if hasattr(logistic_model, 'coef_') else None
            })
        except ValueError as e:
            print(f"Error during linear probing for experiment {experiment_id}: {e}")

        # Clustering (K-Means)
        try:
            kmeans_model = KMeans(n_clusters=number_of_classes, random_state=42, n_init='auto')
            cluster_labels = kmeans_model.fit_predict(test_embeddings)
            purity = calculate_purity(true_labels, cluster_labels)

            evaluation_results.append({
                'evaluation_type': 'clustering',
                'metric_name': 'Purity',
                'metric_value': purity,
                'weights': json.dumps(kmeans_model.cluster_centers_.tolist()) if hasattr(kmeans_model, 'cluster_centers_') else None
            })
        except ValueError as e:
            print(f"Error during clustering for experiment {experiment_id}: {e}")
    else:
        print(f"Warning: Inconsistent data or single class for experiment {experiment_id}, skipping evaluation.")

    return evaluation_results

def store_evaluation_metrics(conn, results, experiment_id):
    """Stores the evaluation metrics in the embedding_evaluation_metrics table."""
    cursor = conn.cursor()
    for result in results:
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

        # Fetch experiments one by one
        cursor.execute(f"SELECT * FROM {EXPERIMENTS_TABLE} LIMIT 2000")

        while True:
            row = cursor.fetchone()
            if row is None:
                break

            experiment = dict(zip([column.name for column in cursor.description], row))
            results = evaluate_embeddings(experiment)
            if results:
                store_evaluation_metrics(conn, results, experiment['experiment_id'])
                print(f"Evaluated experiment {experiment['experiment_id']}")

        cursor.close()
        conn.close()
        print("Evaluation process completed.")

    except psycopg2.Error as e:
        print(f"Error connecting to or querying the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    except psycopg2.Error as e:
        print(f"Error connecting to or querying the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()