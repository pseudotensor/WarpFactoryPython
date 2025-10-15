"""
Plot 3+1 decomposition components of a metric tensor

This module provides visualization for the ADM (Arnowitt-Deser-Misner) 3+1
decomposition of spacetime, plotting the lapse function (alpha), shift vector
(beta), and spatial metric (gamma) components as 2D slices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional
from ..core.tensor import Tensor
from ..core.tensor_ops import verify_tensor
from ..metrics.three_plus_one import three_plus_one_decomposer
from .utils import get_slice_data, label_cartesian_axis, redblue
from matplotlib.colors import ListedColormap


def plot_three_plus_one(
    metric: Tensor,
    sliced_planes: Optional[List[int]] = None,
    slice_locations: Optional[List[int]] = None,
    alpha: float = 0.2
) -> List[Figure]:
    """
    Plot the 3+1 decomposition elements of a metric tensor on a sliced plane.

    This function performs the ADM 3+1 decomposition of spacetime and visualizes:
    - alpha: The lapse function (proper time rate)
    - beta_i: The shift vector components (3 plots)
    - gamma_ij: The spatial metric components (6 plots, upper triangular)

    The decomposition splits spacetime into spatial slices evolving in time,
    which is fundamental to numerical relativity and understanding how space
    curves and evolves.

    Parameters
    ----------
    metric : Tensor
        Metric tensor object to decompose and visualize.
        Must be of type "metric" and have "cartesian" coordinates.

    sliced_planes : list of int, optional
        Two-element list specifying which coordinates to slice (1-indexed: 1=t, 2=x, 3=y, 4=z).
        Example: [1, 4] means slice along t and z, showing the x-y plane.
        Default is [1, 4] (x-y plane).

    slice_locations : list of int, optional
        Two-element list specifying the index locations for the sliced coordinates.
        Default is the center of each sliced dimension.

    alpha : float, optional
        Alpha value for surface grid display transparency from 0 (transparent)
        to 1 (opaque). Default is 0.2.

    Returns
    -------
    list of matplotlib.figure.Figure
        List of Figure objects containing:
        - 1 plot for alpha (lapse function)
        - 3 plots for beta_i (shift vector components)
        - 6 plots for gamma_ij (spatial metric components)
        Total: 10 figures

    Raises
    ------
    ValueError
        If metric type is not "metric"
        If metric is not verified
        If selected planes are the same
        If slice locations are outside the tensor grid bounds
        If coordinate system is not "cartesian"

    Examples
    --------
    Plot 3+1 decomposition in the x-y plane at the center:
    >>> figures = plot_three_plus_one(metric_tensor)

    Plot with custom slice and higher transparency:
    >>> figures = plot_three_plus_one(metric_tensor, alpha=0.5,
    ...                                sliced_planes=[2, 3],
    ...                                slice_locations=[10, 15])

    Notes
    -----
    - The 3+1 decomposition is also known as the ADM formalism
    - The lapse function alpha describes how proper time relates to coordinate time
    - The shift vector beta describes how spatial coordinates move with time
    - The spatial metric gamma describes the geometry of each spatial slice
    - Only works with Cartesian coordinates
    - Uses red-blue diverging colormap centered at zero for all plots
    """
    # Handle default input arguments
    if sliced_planes is None:
        sliced_planes = [1, 4]  # Default: X-Y plane (slice along t and z)

    if slice_locations is None:
        # Get shape from tensor
        s = metric.shape
        # Calculate centers for the sliced planes (1-indexed)
        slice_centers = [round((s[i] + 1) / 2) for i in range(4)]
        slice_locations = [
            slice_centers[sliced_planes[0] - 1],
            slice_centers[sliced_planes[1] - 1]
        ]

    # Check that tensor is a metric
    if metric.type.lower() != "metric":
        raise ValueError("Must provide a metric object.")

    # Verify tensor
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    # Check that the sliced planes are different
    if sliced_planes[0] == sliced_planes[1]:
        raise ValueError("Selected planes must not be the same, select two different planes to slice along.")

    # Round slice locations
    slice_locations = [round(loc) for loc in slice_locations]

    # Check that the slice locations are inside the world
    s = metric.shape
    if (slice_locations[0] < 1 or slice_locations[1] < 1 or
        slice_locations[0] > s[sliced_planes[0] - 1] or
        slice_locations[1] > s[sliced_planes[1] - 1]):
        raise ValueError(
            f"sliceLocations {slice_locations} are outside the world. "
            f"Valid ranges: [1, {s[sliced_planes[0] - 1]}] for plane {sliced_planes[0]}, "
            f"[1, {s[sliced_planes[1] - 1]}] for plane {sliced_planes[1]}"
        )

    # Check coordinate system
    if metric.coords.lower() != "cartesian":
        raise ValueError('Unknown coordinate system, must be: "cartesian"')

    # Perform 3+1 decomposition
    alpha_lapse, beta_down, gamma_down, beta_up, gamma_up = three_plus_one_decomposer(metric)

    # Get axis labels (convert to 0-indexed)
    sliced_planes_0indexed = [sliced_planes[0] - 1, sliced_planes[1] - 1]
    xlabel_text, ylabel_text = label_cartesian_axis(sliced_planes_0indexed)

    # Get slice data indices (convert to 0-indexed)
    slice_locations_0indexed = [slice_locations[0] - 1, slice_locations[1] - 1]
    idx = get_slice_data(sliced_planes_0indexed, slice_locations_0indexed, metric)

    # List to store all figures
    figures = []

    # Plot alpha (lapse function)
    title_text = r"$\alpha$"
    slice_2d = _extract_slice(alpha_lapse, idx)
    fig = _plot_component(slice_2d, title_text, xlabel_text, ylabel_text, alpha)
    figures.append(fig)

    # Plot beta (shift vector) - 3 components
    for i in range(3):
        title_text = rf"$\beta_{{{i+1}}}$"
        slice_2d = _extract_slice(beta_down[i], idx)
        fig = _plot_component(slice_2d, title_text, xlabel_text, ylabel_text, alpha)
        figures.append(fig)

    # Plot gamma (spatial metric) - 6 unique components (symmetric)
    # Components: (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    c = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    for i, j in c:
        # Use 1-indexed for display
        title_text = rf"$\gamma_{{{i+1}{j+1}}}$"
        slice_2d = _extract_slice(gamma_down[(i, j)], idx)
        fig = _plot_component(slice_2d, title_text, xlabel_text, ylabel_text, alpha)
        figures.append(fig)

    return figures


def _extract_slice(array: np.ndarray, idx: List) -> np.ndarray:
    """
    Extract a 2D slice from a 4D array using index specification.

    Parameters
    ----------
    array : numpy.ndarray
        4D array to slice from.
    idx : list
        Four-element list containing index ranges or fixed indices.

    Returns
    -------
    numpy.ndarray
        2D slice with transposed orientation to match MATLAB display.
    """
    # Build the indexing tuple based on which dimensions are sliced
    index_tuple = []
    for dim in range(4):
        if isinstance(idx[dim], int):
            index_tuple.append(idx[dim])
        else:
            index_tuple.append(slice(None))

    # Extract the 2D slice
    slice_2d = array[tuple(index_tuple)]

    # Transpose to match MATLAB's display orientation
    slice_2d = slice_2d.T

    return slice_2d


def _plot_component(
    array: np.ndarray,
    title_text: str,
    xlabel_text: str,
    ylabel_text: str,
    alpha: float = 0.2
) -> Figure:
    """
    Plot a single 3+1 component as a 2D surface.

    Parameters
    ----------
    array : numpy.ndarray
        2D array containing the component values to plot.

    title_text : str
        Title for the plot, typically in LaTeX format (e.g., r"$\alpha$").

    xlabel_text : str
        Label for the x-axis (e.g., "x", "y", "t").

    ylabel_text : str
        Label for the y-axis (e.g., "x", "y", "t").

    alpha : float, optional
        Transparency for grid edges (0=transparent, 1=opaque). Default is 0.2.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot.

    Notes
    -----
    - Uses red-blue diverging colormap centered at zero
    - Displays in top-down view (similar to MATLAB's view(2))
    - White background is used for figure and axes
    - Colorbar is added to show the data range
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # Create the colormap for this data
    cmap_array = redblue(array)
    cmap = ListedColormap(cmap_array)

    # Plot as 2D image with proper extent
    im = ax.imshow(
        array,
        cmap=cmap,
        aspect='auto',
        origin='lower',
        extent=[0, array.shape[1] - 1, 0, array.shape[0] - 1],
        interpolation='nearest'
    )

    # Add grid lines if alpha > 0
    if alpha > 0:
        # Add grid
        ax.set_xticks(np.arange(array.shape[1]))
        ax.set_yticks(np.arange(array.shape[0]))
        ax.grid(True, color='black', alpha=alpha, linewidth=0.5)

    # Labels and title
    ax.set_title(title_text, fontsize=14)
    ax.set_xlabel(xlabel_text, fontsize=12)
    ax.set_ylabel(ylabel_text, fontsize=12)

    # Set limits
    ax.set_xlim([0, array.shape[1] - 1])
    ax.set_ylim([0, array.shape[0] - 1])

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    return fig
