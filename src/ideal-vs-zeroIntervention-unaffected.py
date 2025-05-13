import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from utils import connect_to_db, flatten_extend, fetch_baseline_metric_value, fetch_corrupted_metric_values


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
    ('knn similarity', 'Avg Cosine Similarity (k=1) (all)'),
    ('knn similarity', 'Avg Cosine Similarity (k=3) (all)'),
    ('knn similarity', 'Avg Cosine Similarity (k=5) (all)'),
    ('knn similarity', 'Avg Cosine Similarity (k=10) (all)'),
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
    fetch_baseline_metric_value(conn, evaluation_type=evaluation_type, metric_name=metric_name)
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

        line, = ax.plot(x_values_for_error_type_in_subplot, y_values_for_error_type_in_subplot,
                        marker=markers[error_types[line_idx]],
                        color=colors[error_types[line_idx]],
                        linestyle='-',
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
                           markersize=10, label='perfect data')
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

plt.savefig("ideal-vs-zeroIntervention.eps", dpi=300, bbox_inches='tight', format='eps')
# plt.show()
