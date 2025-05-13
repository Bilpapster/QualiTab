import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines

def fetch_baseline_metric_value(conn, evaluation_type: str = None, metric_name: str = None, tag: str = 'CLEAN_CLEAN'):
    cursor = conn.cursor()
    query = f"""
    SELECT AVG(metric_value) AS baseline_metric_value
    FROM embeddings_experiments ex JOIN embedding_evaluation_metrics ev
    ON ex.experiment_id = ev.experiment_id
    WHERE ex.tag = '{tag}'
    """
    if evaluation_type:
        query += f" AND ev.evaluation_type = '{evaluation_type}'"
    if metric_name:
        query += f" AND ev.metric_name = '{metric_name}'"

    cursor.execute(query)
    return cursor.fetchall()[0][0]

def fetch_corrupted_metric_values(conn, evaluation_type: str = None, metric_name: str = None, tag: str = 'DIRTY_DIRTY'):
    cursor = conn.cursor()
    query = f"""
    SELECT CONCAT(ex.error_type, '_', ex.row_corruption_percent) as type_rate, AVG(metric_value) AS avg_metric_value_error
    FROM embeddings_experiments ex JOIN embedding_evaluation_metrics ev
    ON ex.experiment_id = ev.experiment_id
    WHERE ex.tag = '{tag}'
    """
    if evaluation_type:
        query += f" AND ev.evaluation_type = '{evaluation_type}'"
    if metric_name:
        query += f" AND ev.metric_name = '{metric_name}'"
    query += " GROUP BY type_rate"
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=columns)
    cursor.close()
    # Split the type_rate into error_type and corruption_rate
    df['error_type'] = df['type_rate'].str.split('_').str[0]
    df['corruption_rate'] = df['type_rate'].str.split('_').str[1].astype(int)

    # Convert avg_metric_value_error to float
    df['avg_metric_value_error'] = df['avg_metric_value_error'].astype(float)

    # Drop the original type_rate column and rename avg_metric_value_error to metric_value
    df = df.drop(columns=['type_rate'])
    df = df.rename(columns={'avg_metric_value_error': 'metric_value'})

    # Sort the DataFrame by error_type and (secondly) corruption_rate within each error_type
    result = {}
    for error_type, group in df.groupby('error_type'):
        # Sort again to ensure order is maintained
        group = group.sort_values('corruption_rate')
        result[error_type] = {
            'rates': group['corruption_rate'].tolist(),
            'values': group['metric_value'].tolist()
        }
    return result

# --- Configuration ---
num_plots = 6
num_lines_per_plot = 3
num_points = 3 # Number of points for each line
x_values = np.arange(num_points)

# Define consistent colors (publication-friendly, no green) and markers
# Using seaborn muted palette colors (blue, orange, purple)
colors = ['#4878d0', '#ee854a', '#9d6acc']
markers = ['o', 's', '^']
line_labels = ['Type A', 'Type B', 'Type C'] # Labels for the data lines in the legend

# --- Data Generation ---
# Create random y-data for each plot and each line
# Ensure data is somewhat within the 0-1 range for y-limits, but can vary
np.random.seed(42) # for reproducible random data
all_plot_data = []
for _ in range(num_plots):
    plot_lines_data = [np.random.rand(num_points) * 0.6 + 0.3 for _ in range(num_lines_per_plot)] # data mostly between 0.3 and 0.9
    all_plot_data.append(plot_lines_data)

# Random y-values for the horizontal green dashed line in each plot
horizontal_line_yvals = np.random.rand(num_plots) * 0.5 + 0.4 # Random values between 0.4 and 0.9

# --- Plotting ---
# Create the figure and axes grid
fig, axs = plt.subplots(1, num_plots, figsize=(16, 4), sharey=False) # Adjust figsize as needed

# Ensure axs is always iterable, even if num_plots is 1
if num_plots == 1:
    axs = [axs]

# Store handles for the legend later (only need from the first plot)
legend_handles = []

# Iterate through each subplot axis
for i, ax in enumerate(axs):
    min_y_in_plot = float('inf') # Keep track of min y-value for this specific plot

    # Plot the 3 data lines
    for line_idx in range(num_lines_per_plot):
        y_values = all_plot_data[i][line_idx]
        min_y_in_plot = min(min_y_in_plot, np.min(y_values)) # Update minimum y

        line, = ax.plot(x_values, y_values,
                        marker=markers[line_idx],
                        color=colors[line_idx],
                        linestyle='-',
                        label=line_labels[line_idx]) # Add label for legend handle creation

        # Store handles only from the first plot for the legend
        if i == 0:
            legend_handles.append(line)

    # Plot the horizontal dashed green line
    ax.axhline(y=horizontal_line_yvals[i], color='green', linestyle='--', linewidth=1.5)

    # --- Axis Styling ---
    # Set y-axis limits
    ax.set_ylim(min_y_in_plot - 0.3, 1.0)

    # Set x-axis limits (+-5 from min/max x_values)
    ax.set_xlim(np.min(x_values) - 5, np.max(x_values) + 5)

    # Set x-ticks to only show actual data points
    ax.set_xticks(x_values)
    # Optional: Rotate x-tick labels if they overlap
    # ax.tick_params(axis='x', rotation=45)

    # Add grid lines
    ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5, color='grey')

    # Customize spines (remove top and right frame lines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)   # Keep left spine
    ax.spines['bottom'].set_visible(True) # Keep bottom spine

    # Add title below the plot (using xlabel position)
    ax.set_xlabel(f"Dummy Title {i+1}", labelpad=15) # labelpad adds space between axis and title


# --- Shared Legend ---
# Create dummy lines for the legend items that are not plotted data
ideal_line = mlines.Line2D([], [], color='green', linestyle='--', marker=None,
                           markersize=10, label='ideal')
perfect_line = mlines.Line2D([], [], color='black', linestyle='-', marker=None,
                             markersize=10, label='perfect context')

# Add these dummy handles to our list
all_legend_handles = legend_handles + [ideal_line, perfect_line]

# Create the legend below the subplots
# Adjust bbox_to_anchor y-value (e.g., -0.2, -0.3) to position it correctly below the titles
# The number of columns (ncols) should match the number of legend items for a horizontal layout
fig.legend(handles=all_legend_handles,
           loc='lower center',           # Position center align at the bottom
           bbox_to_anchor=(0.5, -0.25),  # Adjust Y value (-0.25) to control distance below plots
           ncol=len(all_legend_handles), # Number of columns = number of items
           frameon=False)                # Remove legend frame

# --- Final Adjustments ---
# Adjust layout to prevent overlap. May need tweaking depending on figure size and content.
# Using subplots_adjust might be better than tight_layout when placing a fig.legend manually.
fig.subplots_adjust(bottom=0.3) # Increase bottom margin to make space for titles and legend


plt.savefig("publication_friendly_plot.png", dpi=300, bbox_inches='tight') # Save the figure
plt.show()