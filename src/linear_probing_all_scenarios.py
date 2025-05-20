import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from utils import connect_to_db, flatten_extend, fetch_baseline_metric_value, fetch_corrupted_metric_values

ANNOTATIONS_COLOR = '#5B676D' # gnumental gray

# --- Configuration ---
plt.rcParams['font.size'] = '14'
error_types = [
    'MCAR',
    'SCAR',
    'CSCAR',
]  # Error types for the lines

# Using seaborn muted palette colors (blue, orange, purple)
colors = {
    error_types[0]: '#4878d0',
    error_types[1]: '#ee854a',
    error_types[2]: '#9d6acc'
}

markers = {
    error_types[0]: 'o',
    error_types[1]: 's',
    error_types[2]: '^'
}

linestyles = ['-', '--', '-.']  # Different line styles for each error scenario ['zero intervention', 'perfect context']

line_labels = {  # Labels for the data lines in the legend
    error_types[0]: 'missing\nvalues',
    error_types[1]: 'scaling',
    error_types[2]: 'categorical\nshift'
}

evaluation_methods_for_ax_titles = [
    '(a) CLE+TL',
    '(b) COR+TL',
    '(e) ALL+TL',
    '(f) ALL+CC',
]

x_labels = [
    'corruption rate',
    '', # meaning 'corruption rate'
    '',  # meaning 'corruption rate'
    '',  # meaning 'corruption rate'
]

y_labels = [
    'ROC AUC',
    '', # meaning 'ROC AUC'
    '', # meaning 'ROC AUC'
    '',  # meaning 'ROC AUC'
]

evaluation_types_names_for_queries = [
    ('LP_CLE_TLGT', 'ROC_AUC'),
    ('LP_COR_TLGT', 'ROC_AUC'),
    ('LP_ALL_TLGT', 'ROC_AUC'),
    ('LP_ALL_CCGT', 'ROC_AUC'),
]

conn, cursor = connect_to_db()
zero_intervention_metric_values_dicts = [
    fetch_corrupted_metric_values(
        conn, evaluation_type=evaluation_type,
        metric_name=metric_name,
        tag="DIRTY_DIRTY",
    )
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

perfect_context_metric_values_dicts = [
    fetch_corrupted_metric_values(
        conn, evaluation_type=evaluation_type,
        metric_name=metric_name,
        tag="CLEAN_DIRTY",
    )
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

perfect_data_metric_values_dicts = [
    fetch_corrupted_metric_values(
        conn, evaluation_type=evaluation_type,
        metric_name=metric_name,
        tag="CLEAN_CLEAN",
    )
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

num_plots = len(evaluation_methods_for_ax_titles)  # Number of subplots
num_lines_per_plot = len(error_types)

x_values = [ # x-values are the same for both zero_intervention and perfect context
    [dictionary[error_type]['rates'] for error_type in error_types] for dictionary in zero_intervention_metric_values_dicts
]

y_values_zero_intervention = [
    [dictionary[error_type]['values'] for error_type in error_types] for dictionary in zero_intervention_metric_values_dicts
]

y_values_perfect_context = [
    [dictionary[error_type]['values'] for error_type in error_types] for dictionary in perfect_context_metric_values_dicts
]

y_values_perfect_data = [
    [dictionary[error_type]['values'] for error_type in ['NONE']] for dictionary in perfect_data_metric_values_dicts
]

y_values_for_scenarios = [y_values_zero_intervention, y_values_perfect_context, y_values_perfect_data]

# y-values for the horizontal green dashed line in each plot
horizontal_line_yvals = [
    fetch_baseline_metric_value(conn, evaluation_type=evaluation_type, metric_name=metric_name)
    for evaluation_type, metric_name in evaluation_types_names_for_queries
]

# --- Plotting ---
# Create the figure and axes grid
fig, axs = plt.subplots(1, num_plots, figsize=(16, 2), sharey=False)

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
    for scenario_idx, y_values in enumerate(y_values_for_scenarios):
        y_values_for_subplot = y_values[i]

        # handle differently the perfect data scenario (single line, green, without markers)
        if scenario_idx == 2:

            line, = ax.plot([0, 0.1, 0.2, 0.4] if len(y_values_for_subplot[0]) == 4 else [0.1, 0.2, 0.4],
                            np.array(y_values_for_subplot[0]),
                            marker=None,
                            color="green",
                            linewidth=3,
                            linestyle=linestyles[scenario_idx],
                            label="perfect\ndata")  # Add label for legend handle creation

            if i == 0:
                legend_handles.append(line)

            min_y_in_plot = min(min_y_in_plot, np.min(y_values_for_subplot[0]))  # Update minimum y
            max_y_in_plot = max(max_y_in_plot, np.max(y_values_for_subplot[0]))  # Update maximum y
            continue

        # For every other scenario apart from perfect data, plot the 3 data lines
        for line_idx in range(num_lines_per_plot): # essentially for error type in error types
            x_values_for_error_type_in_subplot = x_values_for_subplot[line_idx]
            y_values_for_error_type_in_subplot = y_values_for_subplot[line_idx]
            min_y_in_plot = min(min_y_in_plot, np.min(y_values_for_error_type_in_subplot))  # Update minimum y

            max_y_in_plot = max(max_y_in_plot, np.max(y_values_for_error_type_in_subplot))  # Update maximum y

            line, = ax.plot(x_values_for_error_type_in_subplot,
                            y_values_for_error_type_in_subplot,
                            marker=markers[error_types[line_idx]],
                            color=colors[error_types[line_idx]],
                            linestyle=linestyles[scenario_idx],
                            label=line_labels[error_types[line_idx]])  # Add label for legend handle creation

            # Store handles only from the first plot for the legend
            if scenario_idx == 0 and i ==0:
                # zero intervention should be the first and perfect context the second scenario
                legend_handles.append(line)

    # Plot the horizontal dashed green line
    # ax.axhline(y=horizontal_line_yvals[i], color='green', linestyle=':', linewidth=1.5)

    # --- Axis Styling ---
    # Set y-axis limits
    lower_ylim = min_y_in_plot - (max_y_in_plot - min_y_in_plot) * 0.1
    upper_ylim = max_y_in_plot + (max_y_in_plot - min_y_in_plot) * 0.1
    ax.set_ylim((lower_ylim, upper_ylim)) # center the y-axis around 0.5 for visually better plot with wider range of y-values

    # Set x-axis limits (+-0.05 from min/max x_values)
    ax.set_xlim(np.min(flatten_extend(x_values_for_subplot)) - 0.05, np.max(flatten_extend(x_values_for_subplot)) + 0.05)

    # Set x-ticks to only show actual data points
    ax.set_xticks(x_values_for_subplot[0])
    y_ticks = [
        lower_ylim + (upper_ylim - lower_ylim) / 4,
        lower_ylim + (upper_ylim - lower_ylim)/2,
        min(upper_ylim, 1)
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
# ideal_line = mlines.Line2D([], [], color='green', linestyle='--', marker=None, markersize=10, label='perfect data')

zero_intervention_line = mlines.Line2D([], [], color='black', linestyle='-', marker=None,
                           markersize=10, label='zero\nintervention')

perfect_context_line = mlines.Line2D([], [], color='black', linestyle='--', marker=None,
                           markersize=10, label='perfect\ncontext')

perfect_line = mlines.Line2D([], [], color='green', linestyle=':', marker=None,
                             markersize=10, label='perfect data (baseline)')

# Add these dummy handles to our list
all_legend_handles = legend_handles + [
    perfect_context_line,
    zero_intervention_line,
    # perfect_line,
]

# Create the legend below the subplots
# Adjust bbox_to_anchor y-value (e.g., -0.2, -0.3) to position it correctly below the titles
# The number of columns (ncols) should match the number of legend items for a horizontal layout
fig.legend(handles=all_legend_handles,
           fontsize="16",
           loc='lower center',  # Position center align at the bottom
           bbox_to_anchor=(0.5, -0.35),  # Adjust Y value (-0.25) to control distance below plots
           ncol=len(all_legend_handles),  # Number of columns = number of items
           frameon=True)  # Remove legend frame

# --- Final Adjustments ---
# Adjust layout to prevent overlap. May need tweaking depending on figure size and content.
# Using subplots_adjust might be better than tight_layout when placing a fig.legend manually.
fig.subplots_adjust(bottom=0.3)  # Increase bottom margin to make space for titles and legend

plt.savefig(f"linear_probing_all.eps", dpi=500, bbox_inches='tight', format='eps')
# plt.show()
