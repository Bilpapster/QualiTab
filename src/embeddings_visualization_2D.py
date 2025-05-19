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


"""
-- query to get the experiment ids fro the best and worst roc auc
SELECT ex.experiment_id
--, metric_value, metric_name, evaluation_type, tag, ev.row_corruption_percent, error_type
FROM embedding_evaluation_metrics ev JOIN embeddings_experiments ex
ON ex.experiment_id = ev.experiment_id
WHERE metric_value > 0
AND evaluation_type LIKE '%LP%ALL%TLGT%'
-- AND ex.tag != 'CLEAN_CLEAN'
-- AND ex.row_corruption_percent > 0
ORDER BY metric_value  -- DESC
LIMIT 10
-- OFFSET 5000
"""

experiments = [
    # worst roc auc
    '0b4b36f6-8714-4360-adc2-235753b95e16',
    'e249f5af-fe1c-40df-b2d9-c9b56732b825',
    '1bfcb4e5-9ba2-438e-af0a-2d021564045d',
    'a7b76a4a-81f3-411b-836e-2085c304db05',
    'cc8ba962-0bb6-4e94-a730-e7d11b9b4de5',
    '11c5de3b-a823-4d8f-a57a-00f72c7fd316',
    '5f2ce6f0-e9d5-4b9c-b0d9-4d5f8694ae18',
    '95d67378-b435-42c8-959a-d93902631b72',
    '7a1103c0-b8f5-432f-9c77-a4de851299ec',
    '7160d5bc-3900-4c9b-afb1-fc088cdbc700',

    # best roc auc
    '81dcef79-6802-4b13-9bb8-afe4030e116a',
    '86156091-8493-47a6-ab37-ce6a60c361de',
    '12b8d001-1904-4453-ad81-42ef01714025',
    '6beba83a-fcc1-43cb-9dc8-a2d669c6f434',
    '4b998d8c-6985-46fc-b895-3356e10639d9',
    '615de3a9-fdd0-49bf-a6af-ef890508ecdb',
    '0a9e55a7-8c79-481f-9b63-ad82613630b3',
    '3d024ebf-99cd-49b1-a1f4-a1583e19becc',
    '73e911b5-98b0-4e80-823b-d22bac25afd2',
    '6af3ade1-cb95-4d5e-8fe2-006ca8c9d09e',

    # best roc auc with offset 5000
    'aeed668c-a286-4dc4-86ed-313387973b43',
    '8064a086-07db-4cc8-87fc-47dbd6a4d3a5',
    'ae39fb3a-3064-4f75-bab3-6d09d72b753f',
    'e27dd2d0-f3de-4d18-b5c8-9f8295f67354',
    '59f61a7c-fb75-49ed-9541-a502e8f9d781',
    'd4ad08a6-ca92-4353-b605-0d77f376a598',
    '008a363e-61a7-4321-b9d1-9804d9ebd622',
    'a75a1161-33e7-438a-ae39-da4feab40a43',
    '166ba2b5-4e7f-4303-a655-1ab22be35b97',
    '80b0c9a8-4886-4ad8-a25d-83c64a42cb61',
]

ALPHA = 0.8 # transparency for the points of the embeddings
size = 80
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
    print(f"Working in experiment id {experiment}")
    query = f"""
    SELECT test_embeddings, corrupted_rows, random_seed, dataset_name, tag, error_type, row_corruption_percent
    FROM embeddings_experiments
    WHERE experiment_id = '{experiment}'
    """

    conn, cursor = connect_to_db()
    cursor.execute(query)

    result = fetch_all_as_list_of_dicts(cursor)[0]
    cursor.close()

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

    conn, cursor = connect_to_db()
    get_clean_clean_query = f"""
        SELECT test_embeddings
        FROM embeddings_experiments
        WHERE dataset_name = '{dataset_name}'
        AND tag = 'CLEAN_CLEAN'
        AND error_type = 'NONE'
        AND random_seed = {random_seed}
    """
    cursor.execute(get_clean_clean_query)
    clean_embeddings = cursor.fetchall()[0][0]


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

    unique_labels = np.unique(experiment_object.y_test)
    # map each label to an integer for visualization purposes
    label_mapping = {label: i for i, label in enumerate(unique_labels)}


    edge_colors_left = [colors[label_mapping[label]] for label in experiment_object.y_test]
    edge_colors_right = ['red' if i in corrupted_rows else DARK_GRAY for i in range(len(embeddings))]
    fill_colors_right = [RED_ALPHA_0_5 if i in corrupted_rows else TRANSPARENT_FILL for i in range(len(embeddings))]

    fig, axs = plt.subplots(1, 4, figsize=(20, 4), sharey=False)

    fig.suptitle(f"{experiment} - {str(experiment)} - seed {random_seed} - tag {tag} - {error_type} - {row_corruption_percent}%")


    for i, label in enumerate(unique_labels):
        axs[0].scatter(
            raw_clean_2d[experiment_object.y_test == label, 0], raw_clean_2d[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=size, marker=shapes[i]
        )
        axs[1].scatter(
            clean_embeddings2d[experiment_object.y_test == label, 0], clean_embeddings2d[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=size, marker=shapes[i]
        )
        axs[2].scatter(
            embeddings2d[experiment_object.y_test == label, 0], embeddings2d[experiment_object.y_test == label, 1],
            edgecolors=colors[i], color=fillcolor, s=size, marker=shapes[i]
        )
        axs[3].scatter(
            embeddings2d[experiment_object.y_test == label, 0], embeddings2d[experiment_object.y_test == label, 1],
            edgecolors=[clr for clr,to_add in zip(edge_colors_right, experiment_object.y_test == label) if to_add],
            c=[clr for clr,to_add in zip(fill_colors_right, experiment_object.y_test == label) if to_add],
            s=size, marker=shapes[i]
        )

    axs[0].set_title("Raw data")
    axs[1].set_title("Clean Embeddings")
    axs[2].set_title("Imperfect Embeddings")
    axs[3].set_title("Corrupted data")


    plt.savefig(f"figures/{str(experiment)} - seed {random_seed} - tag {tag} - {error_type} - {row_corruption_percent}%.png",
                dpi=300, bbox_inches='tight', format='png')
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
