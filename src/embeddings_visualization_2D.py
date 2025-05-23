import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from experiment.OpenMLExperiment import OpenMLExperiment
from utils import (
    connect_to_db,
    fetch_all_as_list_of_dicts,
    keep_corrupted_rows_only_for_test_set,
    get_idx_positions_from_idx_values,
)

from enum import Enum

class Transformation(Enum):
    FLIP_HOR = 'FLIP_HOR'
    FLIP_VER = 'FLIP_VER'
    FLIP_BOTH = 'FLIP_BOTH'

    def transform(self, two_dimension_array):
        if self == Transformation.FLIP_HOR or self == Transformation.FLIP_BOTH:
            two_dimension_array[:, 0] = max(two_dimension_array[:, 0]) - two_dimension_array[:, 0]
        if self == Transformation.FLIP_VER or self == Transformation.FLIP_BOTH:
            two_dimension_array[:, 1] = max(two_dimension_array[:, 1]) - two_dimension_array[:, 1]
        return two_dimension_array

RAW = 'RAW'
CLEAN = 'CLEAN'
COR_10 = 'COR_10'
COR_20 = 'COR_20'
COR_40 = 'COR_40'
COR_40_CC = 'COR_40_CC'

experiments = [
    {
        'id': 'd3991990-7ab7-48dc-a16c-bad2bc16e1ec',
        'transformations': {
            COR_40: Transformation.FLIP_VER,
            COR_40_CC: Transformation.FLIP_VER,
        },
        'row_label': 'Cat. shift'
    },
    {
        'id': 'f8ce0b0a-45bb-4cd1-bc97-eb21de5f6f07',
        'transformations': {
            CLEAN: Transformation.FLIP_VER,
            COR_20: Transformation.FLIP_BOTH,
        },
        'row_label': 'Scaling'
    },
    {
        'id': 'b26ec686-719a-4791-8a04-4e3e515748a5',
        'transformations': {
                COR_40: Transformation.FLIP_VER,
                COR_40_CC: Transformation.FLIP_VER,
        },
        'row_label': 'Miss. values'
    },
]

COLUMN_TITLES = [
    '(a) Raw (TL)',
    '(b) Perfect emb. (TL)',
    '(c) 10% corr (TL).',
    '(d) 20% corr. (TL)',
    '(e) 40% corr. (TL)',
    '(f) 40% corr. (CC)',
]

ALPHA = 0.8 # transparency for the points of the embeddings
MARKER_SIZE = 20
FIG_SIZE = (18, 6)
FONT_SIZE = 16
AXES_COLOR = 'darkgray'
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

shapes = ['o', '^', 'X', 'D', 'v', 's', '*', 'P', 'H', '>']

fig, axes = plt.subplots(len(experiments), 6, figsize=FIG_SIZE, sharey=False)
plt.subplots_adjust(wspace=0, hspace=0.04) # adjust the space between plots so that y_labels are more readable


for experiment_index, experiment in enumerate(experiments):
    experiment_id = experiment['id']
    transformations = experiment.get('transformations', None)
    row_label = experiment.get('row_label', None)


    query = f"""
    SELECT test_embeddings, corrupted_rows, random_seed, dataset_name, tag, error_type, row_corruption_percent
    FROM embeddings_experiments
    WHERE experiment_id = '{experiment_id}'
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
    embeddings = embeddings[:len(experiment_object.X_test)]
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
        try:
            corrupted_embeddings[corruption_percent] = np.array(fetch_all_as_list_of_dicts(cursor)[0]['test_embeddings'])
            corrupted_embeddings[corruption_percent] = corrupted_embeddings[corruption_percent][:len(experiment_object.X_test)]
        except IndexError:
            corrupted_embeddings[corruption_percent] = np.array([])


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
    clean_embeddings = clean_embeddings[:len(experiment_object.X_test)]
    cursor.close()

    # Instantiate tsne or PCA
    # lower_dim = TSNE(random_state=random_seed, max_iter=10000, metric="cosine")
    lower_dim = PCA(n_components=2)

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
    if transformations and RAW in transformations:
        raw_clean_2d = transformations[RAW].transform(raw_clean_2d)

    try:
        clean_embeddings2d = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(clean_embeddings)),
        )
        if transformations and CLEAN in transformations:
            clean_embeddings2d = transformations[CLEAN].transform(clean_embeddings2d)

        corrupted_embeddings_10 = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[10])),
        )
        if transformations and COR_10 in transformations:
            corrupted_embeddings_10 = transformations[COR_10].transform(corrupted_embeddings_10)

        corrupted_embeddings_20 = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[20])),
        )
        if transformations and COR_20 in transformations:
            corrupted_embeddings_20 = transformations[COR_20].transform(corrupted_embeddings_20)

        corrupted_embeddings_40 = scaler.fit_transform(
            lower_dim.fit_transform(scaler.fit_transform(corrupted_embeddings[40])),
        )
        if transformations and COR_40 in transformations:
            corrupted_embeddings_40 = transformations[COR_40].transform(corrupted_embeddings_40)
    except ValueError:
        # If the embeddings are empty, skip this experiment
        print(f"Skipping experiment {experiment_id} because the embeddings are empty")
        continue

    # map each label to an integer for visualization purposes
    unique_labels = np.unique(experiment_object.y_test)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    edge_colors_left = [colors[label_mapping[label]] for label in experiment_object.y_test]
    edge_colors_right = ['red' if i in corrupted_rows else DARK_GRAY for i in range(len(embeddings))]
    fill_colors_right = [RED_ALPHA_0_5 if i in corrupted_rows else TRANSPARENT_FILL for i in range(len(embeddings))]

    axs = axes[experiment_index]
    for i, label in enumerate(unique_labels):
        try:
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

            axs[0].set_ylabel(row_label, fontsize=FONT_SIZE, labelpad=10)

        except IndexError:
            # If there are more labels than colors, skip the extra labels
            continue



    for ax_index, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_color(AXES_COLOR)
        ax.spines['right'].set_color(AXES_COLOR)
        ax.spines['left'].set_color(AXES_COLOR)  # Keep left spine
        ax.spines['bottom'].set_color(AXES_COLOR)  # Keep bottom spine
        if experiment_index == 0:
            ax.set_title(COLUMN_TITLES[ax_index], fontsize=FONT_SIZE)

plt.savefig(f"figures/grid.pdf",dpi=100, bbox_inches='tight', format='pdf')
# plt.show()
plt.close()
exit()
