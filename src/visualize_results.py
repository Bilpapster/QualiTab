import psycopg2
import pandas as pd
import numpy as np
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

def fetch_weights_data(conn, evaluation_type, tag=None, error_type=None):
    """
    Fetches weights data from the database for a specific linear probing evaluation type.
    """
    conn, cursor = connect_to_db()
    query = f"""
        SELECT
            e.dataset_name,
            e.tag,
            e.error_type,
            e.row_corruption_percent,
            eem.weights
        FROM
            {EXPERIMENTS_TABLE} e
        JOIN
            {EVALUATION_TABLE} eem ON e.experiment_id = eem.experiment_id
        WHERE
            eem.evaluation_type = %s
            AND eem.weights IS NOT NULL
    """
    params = [evaluation_type]
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

def plot_weights_heatmap_grid(df, evaluation_type_str):
    """
    Generates a grid of heatmaps visualizing the weights of the linear classifier,
    faceted by error type and scenario.
    """
    def process_weights(row):
        # weights = json.loads(row['weights'])
        weights = row['weights']
        # Handle different shapes of weight arrays (binary vs. multi-class)
        if isinstance(weights[0], list):
            return np.array(weights)
        else:
            return np.array([weights]) # Make it 2D for consistent plotting

    df['weights_array'] = df.apply(process_weights, axis=1)

    num_embeddings = df['weights_array'].iloc[0].shape[1] if not df.empty and df['weights_array'].iloc[0].ndim > 1 else None
    if num_embeddings is None:
        print(f"Warning: No weights to visualize for {evaluation_type_str}.")
        return

    g = sns.FacetGrid(df, col='error_type', row='tag', height=3, aspect=1.5, sharex=True, sharey=True)

    def heatmap_on_grid(data, color, **kwargs):
        if not data.empty:
            weights = data['weights_array'].iloc[0]
            sns.heatmap(weights, cmap='coolwarm', cbar_kws={'label': 'Weight Value'}, **kwargs)
            plt.yticks(np.arange(weights.shape[0]) + 0.5, labels=[f'Class {i+1}' for i in range(weights.shape[0])], rotation=0)
            plt.xticks([]) # Remove x-axis ticks for better readability in grid

    g.map_dataframe(heatmap_on_grid, linewidth=0.5, linecolor='lightgray')
    g.set_titles(row_template="Scenario: {row_name}", col_template="Error Type: {col_name}")
    g.set_ylabels("Class")
    g.fig.suptitle(f'Linear Classifier Weights ({evaluation_type_str}) by Error Type and Scenario', y=1.02)
    g.tight_layout()
    plt.show()

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

    # Visualize weights for linear probing on all test data
    weights_df_all = fetch_weights_data(conn, evaluation_type='linear probing fit to all')
    if not weights_df_all.empty:
        plot_weights_heatmap_grid(weights_df_all, 'Fit to All Test Data')

    # Visualize weights for linear probing (unaffected -> corrupted)
    weights_df_unaffected_corrupted = fetch_weights_data(conn, evaluation_type='linear probing unaffected-corrupted')
    if not weights_df_unaffected_corrupted.empty:
        plot_weights_heatmap_grid(weights_df_unaffected_corrupted, 'Fit Unaffected, Predict Corrupted')

    # Visualize weights for linear probing (corrupted -> unaffected)
    weights_df_corrupted_unaffected = fetch_weights_data(conn, evaluation_type='linear probing corrupted-unaffected')
    if not weights_df_corrupted_unaffected.empty:
        plot_weights_heatmap_grid(weights_df_corrupted_unaffected, 'Fit Corrupted, Predict Unaffected')

    conn.close()

if __name__ == "__main__":
    main()
