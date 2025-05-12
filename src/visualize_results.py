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
    plt.legend(title='Error Type & Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_metric_faceted_by_error(df, metric_name, title_prefix=""):
    """
    Generates a grid of line plots of a metric across different corruption rates,
    faceted by error type.
    """
    g = sns.relplot(
        data=df,
        x='row_corruption_percent',
        y='metric_value',
        hue='tag',
        style='tag',
        col='error_type',
        marker='o',
        kind='line',
        col_wrap=3,  # Adjust for the number of columns in the grid
        height=4,
        aspect=1.2
    )
    g.fig.suptitle(f'{title_prefix} {metric_name} vs. Row Corruption (by Error Type)', y=1.02)
    g.set_axis_labels("Row Corruption Percentage", metric_name)
    g.set_titles("Error Type: {col_name}")
    g.add_legend(title='Scenario')
    g.tight_layout()
    plt.show()

def main():
    cursor, conn = connect_to_db()

    # Example: Fetch ROC AUC data for linear probing (all test data)
    roc_auc_df_all = fetch_evaluation_data(conn, metric_name='ROC AUC', evaluation_type='linear probing fit to all')
    if not roc_auc_df_all.empty:
        plot_metric_across_corruption(roc_auc_df_all, metric_name='ROC AUC (Linear Probing - All)', title_prefix="Average")
        plot_metric_faceted_by_error(roc_auc_df_all, metric_name='ROC AUC (Linear Probing - All)')

    # Fetch Purity data for clustering (all test data)
    purity_df_all = fetch_evaluation_data(conn, metric_name='Purity', evaluation_type='clustering all test')
    if not purity_df_all.empty:
        plot_metric_across_corruption(purity_df_all, metric_name='Purity (Clustering - All)', title_prefix="Average")
        plot_metric_faceted_by_error(purity_df_all, metric_name='Purity (Clustering - All)')

    # Fetch Average Cosine Similarity data for KNN (k=5, all test data)
    knn_df_k5_all = fetch_evaluation_data(conn, metric_name='Avg Cosine Similarity (k=5) (all)', evaluation_type='knn similarity')
    if not knn_df_k5_all.empty:
        plot_metric_across_corruption(knn_df_k5_all, metric_name='Avg Cosine Similarity (k=5 - All)', title_prefix="Average")
        plot_metric_faceted_by_error(knn_df_k5_all, metric_name='Avg Cosine Similarity (k=5 - All)')

    conn.close()

if __name__ == "__main__":
    main()
