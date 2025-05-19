import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import openml
import pprint
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from experiment.OpenMLExperiment import OpenMLExperiment
from utils import (
    connect_to_db,
    flatten_extend,
    fetch_baseline_metric_value,
    fetch_corrupted_metric_values,
    fetch_all_as_list_of_dicts,
    keep_corrupted_rows_only_for_test_set,
    get_idx_positions_from_idx_values,
)

from experiments_to_visualize import experiments

ALPHA = 0.8 # transparency for the points of the embeddings
MARKER_SIZE = 80
FIG_SIZE = (20, 3)
# Fill and edge colors:
TRANSPARENT_FILL = "#FFFFFF00" # white color with alpha=0 (00 in hex)
RED_ALPHA_0_5 = "#FF000080" # red color with alpha=0.5 (80 in hex)
DARK_GRAY = "#121212" # dark gray color for unaffected rows


# 10 different colors from the seaborn color palette. We exclude red to avoid confusion with the corrupted rows
colors = ['#4878D0', '#EE854A', '#6ACC64', '#9d6acc', '#956CB4', '#8C613C', '#DC7EC0', '#797979', '#D5BB67', '#82C6E2']
colors = [color.lstrip('#') for color in colors] # remove the '#' from the hex color codes
colors = [tuple(int(color[i:i+2], 16)/255 for i in (0, 2, 4)) for color in colors] # transform to RGB
colors = [tuple(list(color) + [ALPHA]) for color in colors] # add alpha channel

fillcolor = "#FFFFFF00" # white color with alpha=0.5 (80 in hex)

# in case you prefer standard colors, you can fall back to this:
# colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'turquoise', 'gray', 'yellow', 'aquamarine']

shapes = ['o', 'X', '^', 'D', 'v', 's', '*', 'P', 'H', '>']

for experiment in experiments:
    query = f"""
    SELECT test_embeddings, corrupted_rows, random_seed, dataset_name, tag, error_type, row_corruption_percent
    FROM embeddings_experiments
    WHERE experiment_id = '{experiment}'
    """

    conn, cursor = connect_to_db()
    cursor.execute(query)

    result = fetch_all_as_list_of_dicts(cursor)[0]

    embeddings = np.array(result['test_embeddings'])
    corrupted_rows = keep_corrupted_rows_only_for_test_set(result['corrupted_rows'])
    random_seed = result['random_seed']
    dataset_name = result['dataset_name']
    dataset_id = int(dataset_name.split('-')[1])
    tag = result['tag']
    error_type = result['error_type']
    row_corruption_percent = result['row_corruption_percent']

    experiment_object = OpenMLExperiment()
    experiment_object.random_seed = random_seed
    experiment_object.task = openml.tasks.get_task(OpenMLExperiment.get_task_id_from_dataset_name(dataset_name))
    experiment_object.dataset_id = dataset_id
    experiment_object.load_dataset(dataset_config=dict())
    corrupted_rows = get_idx_positions_from_idx_values(corrupted_rows, experiment_object.X_test)
    unaffected_rows = [i for i in range(len(embeddings)) if i not in corrupted_rows]

    # get the corrupted embeddings for 10%, 20% and 40% corruption
    corrupted_embeddings = dict()
    for corruption_percent in [10, 20, 40]:
        if corruption_percent == row_corruption_percent:
            corrupted_embeddings[corruption_percent] = embeddings
            continue

        get_corrupted_embeddings_query = f"""
            SELECT test_embeddings
            FROM embeddings_experiments
            WHERE dataset_name = '{dataset_name}'
            AND tag = '{tag}'
            AND error_type = '{error_type}'
            AND random_seed = {random_seed}
            AND row_corruption_percent = {corruption_percent}
        """
        cursor.execute(get_corrupted_embeddings_query)
        corrupted_embeddings[corruption_percent] = np.array(fetch_all_as_list_of_dicts(cursor)[0]['test_embeddings'])

    # get the clean embeddings
    get_clean_clean_query = f"""
        SELECT test_embeddings
        FROM embeddings_experiments
        WHERE dataset_name = '{dataset_name}'
        AND tag = 'CLEAN_CLEAN'
        AND error_type = 'NONE'
        AND random_seed = {random_seed}
    """
    cursor.execute(get_clean_clean_query)
    clean_embeddings = np.array(fetch_all_as_list_of_dicts(cursor)[0]['test_embeddings'])
    cursor.close()

    # Instantiate tsne, specify cosine metric
    lower_dim = TSNE(random_state=random_seed, max_iter=10000, metric="cosine")
    # lower_dim = PCA(n_components=2)
    scaler = StandardScaler()
    # Fit and transform
    embeddings2d = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(embeddings)),
    )

    raw_clean = experiment_object.X_test
    raw_clean = raw_clean.apply(lambda x:x.fillna(x.value_counts().index[0])) # fill missing values with most frequent value of that column
    raw_clean = pd.get_dummies(raw_clean, dtype=int) # convert categorical variables to one-hot encoded values
    raw_clean_2d = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(raw_clean)),
    )

    clean_embeddings2d = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(clean_embeddings)),
    )

    corrupted_embeddings_10 = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[10])),
    )
    corrupted_embeddings_20 = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[20])),
    )
    corrupted_embeddings_40 = scaler.fit_transform(
        lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[40])),
    )

    unique_labels = np.unique(experiment_object.y_test)
    # map each label to an integer for visualization purposes
    label_mapping = {label: i for i, label in enumerate(unique_labels)}


    edge_colors_left = [colors[label_mapping[label]] for label in experiment_object.y_test]
    edge_colors_right = ['red' if i in corrupted_rows else DARK_GRAY for i in range(len(embeddings))]
    fill_colors_right = [RED_ALPHA_0_5 if i in corrupted_rows else TRANSPARENT_FILL for i in range(len(embeddings))]

    fig, axs = plt.subplots(1, 6, figsize=FIG_SIZE, sharey=False)

    for i, label in enumerate(unique_labels):
        axs[0].scatter(
            raw_clean_2d[experiment_object.y_test == label, 0], raw_clean_2d[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=MARKER_SIZE, marker=shapes[i]
        )
        axs[1].scatter(
            clean_embeddings2d[experiment_object.y_test == label, 0], clean_embeddings2d[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=MARKER_SIZE, marker=shapes[i]
        )
        axs[2].scatter(
            corrupted_embeddings_10[experiment_object.y_test == label, 0], corrupted_embeddings_10[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=MARKER_SIZE, marker=shapes[i]
        )
        axs[3].scatter(
            corrupted_embeddings_20[experiment_object.y_test == label, 0], corrupted_embeddings_20[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=MARKER_SIZE, marker=shapes[i]
        )
        axs[4].scatter(
            corrupted_embeddings_40[experiment_object.y_test == label, 0], corrupted_embeddings_40[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=MARKER_SIZE, marker=shapes[i]
        )
        axs[5].scatter(
            corrupted_embeddings_40[experiment_object.y_test == label, 0], corrupted_embeddings_40[experiment_object.y_test == label, 1],
            edgecolors=[clr for clr,to_add in zip(edge_colors_right, experiment_object.y_test == label) if to_add],
            c=[clr for clr,to_add in zip(fill_colors_right, experiment_object.y_test == label) if to_add],
            s=MARKER_SIZE, marker=shapes[i]
        )

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(True)  # Keep left spine
        # ax.spines['bottom'].set_visible(True)  # Keep bottom spine

    axs[0].set_title("Clean data")
    axs[1].set_title("Clean embeddings")
    axs[2].set_title("10% corruption")
    axs[3].set_title("20% corruption")
    axs[4].set_title("40% corruption")
    axs[5].set_title("40% corruption (corrupted in red)")


    plt.savefig(f"figures/{error_type} - {random_seed} - {dataset_name}%.png",
                dpi=100, bbox_inches='tight', format='png')
    # plt.show()
    plt.close()
# plt.set_title("Embedded data + PCA")
# plt.set_xlabel("PCA 1")
# plt.set_ylabel("PCA 2")
# plt.set_xticks([])
# plt.set_yticks([])
exit()


query = """
SELECT ev.experiment_id, ev.metric_name, ev.metric_value, ex.tag
FROM embeddings_experiments ex JOIN embedding_evaluation_metrics ev
ON ex.experiment_id = ev.experiment_id
WHERE ex.tag != 'CLEAN_CLEAN'
AND ev.evaluation_type LIKE 'clustering all test' 
-- or ev.evaluation_type LIKE 'linear probing fit to all'
ORDER BY ev.metric_value DESC
LIMIT 10
"""

conn, cursor = connect_to_db()
cursor.execute(query)
rows = cursor.fetchall()
import pprint
pprint.pp(rows)
exit()


exit()


# --- Configuration ---
plt.rcParams['font.size'] = '14'
error_types = ['MCAR', 'SCAR', 'CSCAR']  # Error types for the lines

# Using seaborn muted palette colors (blue, orange, purple)
colors = {
    error_types[0]: '#4878d0',
    error_types[1]: '#ee854a',
    error_types[2]: '#9d6acc'}

markers = {
    error_types[0]: 'o',
    error_types[1]: 's',
    error_types[2]: '^'
}

line_labels = {  # Labels for the data lines in the legend
    error_types[0]: 'missing values',
    error_types[1]: 'scaling',
    error_types[2]: 'categorical shift'
}

evaluation_methods_for_ax_titles = [
    '(a) 1-NN',
    '(b) 3-NN',
    '(c) 5-NN',
    '(d) 10-NN',
    '(e) Linear Probing',
    '(f) K-means',
]

x_labels = [
    'corruption rate',
    '', # meaning 'corruption rate'
    '', # meaning 'corruption rate'
    '', # meaning 'corruption rate'
    '', # meaning 'corruption rate'
    '', # meaning 'corruption rate'
]

y_labels = [
    'Cosine Similarity',
    '', # meaning 'Cosine Similarity',
    '', # meaning'Cosine Similarity',
    '', # meaning'Cosine Similarity',
    'ROC AUC',
    'Purity',
]

evaluation_types_names_for_queries = [
    ('knn similarity', 'Avg Cosine Similarity (k=1) (unaffected)'),
    ('knn similarity', 'Avg Cosine Similarity (k=3) (unaffected)'),
    ('knn similarity', 'Avg Cosine Similarity (k=5) (unaffected)'),
    ('knn similarity', 'Avg Cosine Similarity (k=10) (unaffected)'),
    ('linear probing fit to all', 'ROC AUC'),
    ('clustering all test', 'Purity'),
]

conn, cursor = connect_to_db()
corrupted_metric_values_dicts = [
    fetch_corrupted_metric_values(
        conn, evaluation_type=evaluation_type,
        metric_name=str(metric_name).replace('all', 'unaffected'),
        tag="DIRTY_DIRTY"
    )
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

num_plots = len(evaluation_methods_for_ax_titles)  # Number of subplots
num_lines_per_plot = len(error_types)

x_values = [
    [dictionary[error_type]['rates'] for error_type in error_types] for dictionary in corrupted_metric_values_dicts
]
y_values = [
    [dictionary[error_type]['values'] for error_type in error_types] for dictionary in corrupted_metric_values_dicts
]

# y-values for the horizontal green dashed line in each plot
horizontal_line_yvals = [
    fetch_baseline_metric_value(conn, evaluation_type=evaluation_type, metric_name=metric_name, tag='CLEAN_DIRTY')
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

# --- Plotting ---
# Create the figure and axes grid
fig, axs = plt.subplots(1, num_plots, figsize=(16, 2), sharey=False)
plt.subplots_adjust(wspace=0.66) # adjust the space between plots so that y_labels are more readable

# Ensure axs is always iterable, even if num_plots is 1
if num_plots == 1:
    axs = [axs]

# Store handles for the legend later (only need from the first plot)
legend_handles = []


# Iterate through each subplot axis
for i, ax in enumerate(axs):
    min_y_in_plot = float('inf')  # Keep track of min y-value for this specific plot
    max_y_in_plot = float('-inf')  # Keep track of max y-value for this specific plot
    x_values_for_subplot = np.array(x_values[i], dtype=float) / 100
    y_values_for_subplot = y_values[i]

    # Plot the 3 data lines
    for line_idx in range(num_lines_per_plot): # essentially for error type in error types
        x_values_for_error_type_in_subplot = x_values_for_subplot[line_idx]
        y_values_for_error_type_in_subplot = y_values_for_subplot[line_idx]
        min_y_in_plot = min(min_y_in_plot, np.min(y_values_for_error_type_in_subplot), horizontal_line_yvals[i])  # Update minimum y
        max_y_in_plot = max(max_y_in_plot, np.max(y_values_for_error_type_in_subplot), horizontal_line_yvals[i])  # Update maximum y

        line = ax.scatter(x_values_for_error_type_in_subplot, y_values_for_error_type_in_subplot,
                        marker=markers[error_types[line_idx]],
                        color=colors[error_types[line_idx]],
                        # linestyle='-',
                        label=line_labels[error_types[line_idx]])  # Add label for legend handle creation

        # Store handles only from the first plot for the legend
        if i == 0:
            legend_handles.append(line)

    # Plot the horizontal dashed green line
    ax.axhline(y=horizontal_line_yvals[i], color='green', linestyle=':', linewidth=1.5)

    lower_ylim = min_y_in_plot - (max_y_in_plot - min_y_in_plot) * 0.1
    upper_ylim = max_y_in_plot + (max_y_in_plot - min_y_in_plot) * 0.1
    ax.set_ylim((lower_ylim, upper_ylim))  # center the y-axis around 0.5 for visually better plot with wider range of y-values

    # Set x-axis limits (+-5 from min/max x_values)
    ax.set_xlim(np.min(flatten_extend(x_values_for_subplot)) - 0.05, np.max(flatten_extend(x_values_for_subplot)) + 0.05)

    # Set x-ticks to only show actual data points
    ax.set_xticks(x_values_for_subplot[0])
    y_ticks = [
        lower_ylim + (upper_ylim - lower_ylim) / 4,
        lower_ylim + (upper_ylim - lower_ylim)/2,
        upper_ylim
    ]
    number_of_decimal_places = 2
    if round(y_ticks[0], number_of_decimal_places) == round(y_ticks[1], number_of_decimal_places) or round(y_ticks[1], number_of_decimal_places) == round(y_ticks[2], number_of_decimal_places):
        number_of_decimal_places = 3 # todo find a better way because this is disgusting

    ax.set_yticks([round(tick, number_of_decimal_places) for tick in y_ticks])  # Set y-ticks to be in steps of 0.1
    # Optional: Rotate x-tick labels if they overlap
    # ax.tick_params(axis='x', rotation=45)

    # Add grid lines
    ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5, color='grey')

    # Customize spines (remove top and right frame lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)  # Keep left spine
    ax.spines['bottom'].set_visible(True)  # Keep bottom spine

    # Add title below the plot (using xlabel position) this should be placed above the plot
    ax.set_title(evaluation_methods_for_ax_titles[i], pad=15)  # labelpad adds space between axis and title
    ax.set_xlabel(x_labels[i], labelpad=2)  # labelpad adds space between axis and title
    ax.set_ylabel(y_labels[i], labelpad=2)  # labelpad adds space between axis and title

# --- Shared Legend ---
# Create dummy lines for the legend items that are not plotted data
ideal_line = mlines.Line2D([], [], color='green', linestyle=':', marker=None,
                           markersize=10, label='perfect data (baseline)')
# perfect_line = mlines.Line2D([], [], color='black', linestyle='-', marker=None, markersize=10, label='perfect context')

# Add these dummy handles to our list
all_legend_handles = legend_handles + [ideal_line] # + [ideal_line, perfect_line] if needed

# Create the legend below the subplots
# Adjust bbox_to_anchor y-value (e.g., -0.2, -0.3) to position it correctly below the titles
# The number of columns (ncols) should match the number of legend items for a horizontal layout
fig.legend(handles=all_legend_handles,
           fontsize="16",
           loc='lower center',  # Position center align at the bottom
           bbox_to_anchor=(0.5, -0.25),  # Adjust Y value (-0.25) to control distance below plots
           ncol=len(all_legend_handles),  # Number of columns = number of items
           frameon=True)  # Remove legend frame

# --- Final Adjustments ---
# Adjust layout to prevent overlap. May need tweaking depending on figure size and content.
# Using subplots_adjust might be better than tight_layout when placing a fig.legend manually.
fig.subplots_adjust(bottom=0.3)  # Increase bottom margin to make space for titles and legend

plt.savefig("embeddings2D.eps", dpi=300, bbox_inches='tight', format='eps')
# plt.show()
