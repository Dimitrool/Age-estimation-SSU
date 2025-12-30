import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List

def plot_age_distribution_heatmap(
    ages1: List[float],
    ages2: List[float],
    result_path: str,
    bin_width: int = 5,
    min_age: int = 0,
    max_age: int = 100
):
    """
    Generates a 2D histogram to show the joint distribution of ages.

    Args:
        ages1 (List[float]): A list of the true ages of the first person in each pair.
        ages2 (List[float]): A list of the true ages of the second person in each pair.
        bin_width (int): The width (in years) of the bins for the histogram.
        min_age (int): The minimum age for the axes (default 0).
        max_age (int): The maximum age for the axes (default 100).
    """
    # 1. Define the complete bin structure
    bins = np.arange(min_age, max_age + bin_width, bin_width)

    # 2. Compute the 2D histogram matrix using NumPy
    count_matrix, _, _ = np.histogram2d(ages1, ages2, bins=[bins, bins])
    count_matrix = count_matrix.T

    # 3. **Handle Zeros for Log Scale**
    # A log scale cannot handle a value of 0. We mask these values so they
    # are not passed to the normalizer and will appear as blank cells.
    plot_matrix = np.ma.masked_equal(count_matrix, 0)

    # 4. Plotting the computed matrix with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use imshow with a LogNorm normalizer.
    # The normalizer maps the data values to the 0-1 range for the colormap.
    # The LogNorm does this on a logarithmic scale.
    im = ax.imshow(
        plot_matrix,
        interpolation='nearest',
        origin='lower',
        cmap='inferno',
        extent=[min_age, max_age, min_age, max_age],
        norm=LogNorm() # Use the logarithmic normalizer
    )

    # 5. Styling the plot
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of Pairs (Log Scale)', fontsize=12)

    ax.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Age1 = Age2')

    ax.set_title('2D Heatmap of Paired Ages Distribution (Log Scale)', fontsize=16)
    ax.set_xlabel('Age of Person 1', fontsize=12)
    ax.set_ylabel('Age of Person 2', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set the ticks to match the bin centers
    tick_labels = [f'{i}-{i+bin_width-1}' for i in bins[:-1]]
    ax.set_xticks(bins[:-1] + bin_width / 2)
    ax.set_yticks(bins[:-1] + bin_width / 2)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(f"{result_path}/age_distribution.png", dpi=300)
    

def plot_prediction_error_heatmap(
    ages1: List[float],
    ages2: List[float],
    predicted_ages2: List[float],
    result_path: str,
    bin_width: int = 5
):
    """
    Generates a 2D heatmap showing the mean prediction error.

    Args:
        ages1 (List[float]): True ages of Person 1 in each pair.
        ages2 (List[float]): True ages of Person 2 in each pair.
        predicted_ages2 (List[float]): Model's predicted ages for Person 2.
    """
    
    # 1. Define the complete bin structure for the 0-100 range
    min_age = 0
    max_age = 100
    bins = np.arange(min_age, max_age + bin_width, bin_width)
    num_bins = len(bins) - 1

    # Convert lists to NumPy arrays for efficient processing
    ages1_arr = np.array(ages1)
    ages2_arr = np.array(ages2)
    errors_arr = np.array(predicted_ages2) - ages2_arr

    # 2. Manually compute the mean error matrix
    sum_matrix = np.zeros((num_bins, num_bins))
    count_matrix = np.zeros((num_bins, num_bins))

    # Use np.digitize to find which bin each age falls into. It's much faster than a loop.
    # It returns indices from 1 to N, so we subtract 1 for 0-based indexing.
    x_indices = np.digitize(ages1_arr, bins) - 1
    y_indices = np.digitize(ages2_arr, bins) - 1

    # Loop through each data point to populate the matrices
    for i in range(len(ages1_arr)):
        # Ensure the data point is within our desired 0-100 range
        if 0 <= x_indices[i] < num_bins and 0 <= y_indices[i] < num_bins:
            sum_matrix[y_indices[i], x_indices[i]] += errors_arr[i]
            count_matrix[y_indices[i], x_indices[i]] += 1

    # Calculate the mean error. Use np.errstate to avoid warnings for division by zero.
    # Where count is 0, the result will be NaN.
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_error_matrix = sum_matrix / count_matrix

    # 3. Plotting the computed matrix with Matplotlib
    fig, ax = plt.subplots(figsize=(13, 10))

    # We want higher ages at the top, so we use origin='upper' and flip the matrix.
    # Flipping is more intuitive than changing the plot origin for this case.
    matrix_to_plot = np.flipud(mean_error_matrix)
    
    # Define the color normalization to center at 0
    # Find the maximum absolute error to make the color scale symmetric
    max_abs_error = np.nanmax(np.abs(matrix_to_plot))
    
    im = ax.imshow(
        matrix_to_plot,
        interpolation='nearest',
        cmap='coolwarm',
        vmin=-max_abs_error, # Center the colormap
        vmax=max_abs_error
    )

    # 4. Styling the plot
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean Error (Predicted - True)', fontsize=12)

    ax.set_title('Mean Prediction Error by True Age Pair', fontsize=18)
    ax.set_xlabel('Age of Face 1', fontsize=14)
    ax.set_ylabel('Age of Face 2', fontsize=14)
    ax.set_aspect('equal')

    # Set the ticks and labels
    tick_labels = [f'{i}-{i+bin_width-1}' for i in bins[:-1]]
    ax.set_xticks(np.arange(num_bins))
    ax.set_yticks(np.arange(num_bins))
    ax.set_xticklabels(tick_labels, rotation=90)
    # Since we flipped the matrix, we need to set the y-tick labels in reverse order
    ax.set_yticklabels(tick_labels[::-1])

    # Add white grid lines to separate cells, matching the desired look
    ax.set_xticks(np.arange(num_bins + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_bins + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False) # Hide minor ticks

    plt.tight_layout()
    plt.savefig(f"{result_path}/age_error.png", dpi=300)