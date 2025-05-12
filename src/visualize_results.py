import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import connect_to_db

# Table names
EXPERIMENTS_TABLE = "embeddings_experiments"
EVALUATION_TABLE = "embedding_evaluation_metrics"

def fetch_evaluation_data(conn, metric_name, evaluation_type=None, tag=None, error_type=None):
    """
    Fetches evaluation data from the database based on specified criteria.
    """
    conn, cursor = connect_to_db()
    query = f"""
        SELECT
            e.dataset_name,
            e.tag,
            e.error_type,
            e.row_corruption_percent,
            eem.metric_value
        FROM
            {EXPERIMENTS_TABLE} e
        JOIN
            {EVALUATION_TABLE} eem ON e.experiment_id = eem.experiment_id
        WHERE
            eem.metric_name = %s
    """
    params = [metric_name]
    if evaluation_type:
        query += " AND eem.evaluation_type = %s"
        params.append(evaluation_type)
    if tag:
        query += " AND e.tag = %s"
        params.append(tag)
    if error_type:
        query += " AND e.error_type = %s"
        params.append(error_type)

    cursor.execute(query, tuple(params))
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=columns)
    cursor.close()
    return df

def plot_metric_across_corruption(df, metric_name, title_prefix=""):
    """
    Generates line plots of a metric across different corruption rates, grouped by error type and scenario.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='row_corruption_percent', y='metric_value', hue='error_type', style='tag', marker='o')
    plt.title(f'{title_prefix} {metric_name} vs. Row Corruption')
    plt.xlabel('Row Corruption Percentage')
    plt.ylabel(metric_name)
    plt.legend(title='Error Type & Scenario')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    cursor, conn = connect_to_db()

    # Example: Fetch ROC AUC data for linear probing
    roc_auc_df = fetch_evaluation_data(conn, metric_name='ROC AUC', evaluation_type='linear probing fit to all')
    if not roc_auc_df.empty:
        plot_metric_across_corruption(roc_auc_df, metric_name='ROC AUC (Linear Probing - All)', title_prefix="Average")

    # We will add more plotting functions and calls here

    conn.close()

if __name__ == "__main__":
    main()
