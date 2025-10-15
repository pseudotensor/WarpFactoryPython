"""
Plot tensor components as 2D slices

This module provides the main plotting function for visualizing tensor fields
by extracting 2D slices from 4D spacetime data and displaying each unique
tensor component.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Optional
from ..core.tensor import Tensor
from ..core.tensor_ops import verify_tensor
from .utils import get_slice_data, label_cartesian_axis, redblue
from matplotlib.colors import ListedColormap


def plot_tensor(
    tensor: Tensor,
    alpha: float = 0.2,
    sliced_planes: Optional[List[int]] = None,
    slice_locations: Optional[List[int]] = None
) -> List[Figure]:
    """
    Plot the unique elements of a tensor on a sliced plane.

    This function extracts a 2D slice from a 4D tensor and plots each unique
    tensor component as a separate figure. The slice is taken by fixing two
    coordinates at specified locations and displaying the remaining two.

    Parameters
    ----------
    tensor : Tensor
        Tensor object containing either metric or stress-energy tensor data.
        Must have attributes: tensor (dict), type (str), index (str), coords (str).

    alpha : float, optional
        Alpha value for surface grid display transparency from 0 (transparent)
        to 1 (opaque). Default is 0.2.

    sliced_planes : list of int, optional
        Two-element list specifying which coordinates to slice (1-indexed: 1=t, 2=x, 3=y, 4=z).
        Example: [1, 4] means slice along t and z, showing the x-y plane.
        Default is [1, 4] (x-y plane).

    slice_locations : list of int, optional
        Two-element list specifying the index locations for the sliced coordinates.
        Default is the center of each sliced dimension.

    Returns
    -------
    list of matplotlib.figure.Figure
        List of Figure objects, one for each unique tensor component plotted.

    Raises
    ------
    ValueError
        If tensor is not verified
        If selected planes are the same
        If slice locations are outside the tensor grid bounds

    Examples
    --------
    Plot a metric tensor in the x-y plane at the center:
    >>> figures = plot_tensor(metric_tensor)

    Plot a stress-energy tensor with custom slice:
    >>> figures = plot_tensor(stress_energy, alpha=0.5, sliced_planes=[2, 3],
    ...                       slice_locations=[10, 15])

    Plot with high transparency grid lines:
    >>> figures = plot_tensor(tensor, alpha=0.1)

    Notes
    -----
    - Coordinate indexing uses MATLAB convention (1-indexed: 1=t, 2=x, 3=y, 4=z)
    - For symmetric tensors (covariant/contravariant), only unique components are plotted
    - For mixed index tensors, all 16 components are plotted
    - Each component is displayed with a red-blue diverging colormap centered at zero
    - Figures are created with white backgrounds and 2D top-down view
    """
    # Handle default input arguments
    if sliced_planes is None:
        sliced_planes = [1, 4]  # Default: X-Y plane (slice along t and z)

    if slice_locations is None:
        # Get shape from first tensor component
        s = tensor.shape
        # Calculate centers for the sliced planes (1-indexed)
        slice_centers = [round((s[i] + 1) / 2) for i in range(4)]
        slice_locations = [
            slice_centers[sliced_planes[0] - 1],
            slice_centers[sliced_planes[1] - 1]
        ]

    # Verify tensor
    if not verify_tensor(tensor, suppress_msgs=True):
        raise ValueError("Tensor is not verified. Please verify tensor using verify_tensor(tensor).")

    # Check that the sliced planes are different
    if sliced_planes[0] == sliced_planes[1]:
        raise ValueError("Selected planes must not be the same, select two different planes to slice along.")

    # Round slice locations
    slice_locations = [round(loc) for loc in slice_locations]

    # Check that the slice locations are inside the world
    s = tensor.shape
    if (slice_locations[0] < 1 or slice_locations[1] < 1 or
        slice_locations[0] > s[sliced_planes[0] - 1] or
        slice_locations[1] > s[sliced_planes[1] - 1]):
        raise ValueError(
            f"sliceLocations {slice_locations} are outside the world. "
            f"Valid ranges: [1, {s[sliced_planes[0] - 1]}] for plane {sliced_planes[0]}, "
            f"[1, {s[sliced_planes[1] - 1]}] for plane {sliced_planes[1]}"
        )

    # Determine title character based on tensor type
    if tensor.type.lower() == "metric":
        title_character = "g"
    elif tensor.type.lower() == "stress-energy":
        title_character = "T"
    else:
        raise ValueError(f"Unknown tensor type: {tensor.type}")

    # Determine title formatting based on index type
    if tensor.index.lower() == "covariant":
        title_augment1 = "_{"
        title_augment2 = ""
    elif tensor.index.lower() == "contravariant":
        title_augment1 = "^{"
        title_augment2 = ""
    elif tensor.index.lower() == "mixedupdown":
        title_augment1 = "^{"
        title_augment2 = "}_{"
    elif tensor.index.lower() == "mixeddownup":
        title_augment1 = "_{"
        title_augment2 = "}^{"
    else:
        raise ValueError(f"Unknown index type: {tensor.index}")

    # Check coordinate system
    if tensor.coords.lower() != "cartesian":
        raise ValueError('Unknown coordinate system, must be: "cartesian"')

    # Get axis labels (convert to 0-indexed)
    sliced_planes_0indexed = [sliced_planes[0] - 1, sliced_planes[1] - 1]
    xlabel_text, ylabel_text = label_cartesian_axis(sliced_planes_0indexed)

    # Determine which components to plot based on index type
    if tensor.index.lower() in ["mixedupdown", "mixeddownup"]:
        # Mixed index: plot all 16 components
        # Convert to 0-indexed for Python
        c1 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        c2 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    else:
        # Covariant or contravariant: plot only unique components (upper triangular)
        # Convert to 0-indexed for Python
        c1 = [0, 0, 0, 0, 1, 2, 3, 1, 1, 2]
        c2 = [0, 1, 2, 3, 1, 2, 3, 2, 3, 3]

    # Get slice data indices (convert to 0-indexed)
    slice_locations_0indexed = [slice_locations[0] - 1, slice_locations[1] - 1]
    idx = get_slice_data(sliced_planes_0indexed, slice_locations_0indexed, tensor)

    # Plot each component
    figures = []
    for i in range(len(c1)):
        # Extract the 2D slice
        # We need to handle the indexing carefully for arbitrary slice planes
        component_data = tensor.tensor[(c1[i], c2[i])]

        # Build the indexing tuple based on which dimensions are sliced
        index_tuple = []
        for dim in range(4):
            if isinstance(idx[dim], int):
                index_tuple.append(idx[dim])
            else:
                index_tuple.append(slice(None))

        # Extract the 2D slice
        slice_2d = component_data[tuple(index_tuple)]

        # Transpose to match MATLAB's display orientation
        slice_2d = slice_2d.T

        # Create title with proper index notation
        # Convert back to 1-indexed for display
        title_text = f"${title_character}{title_augment1}{c1[i]+1}{title_augment2}{c2[i]+1}}}$"

        # Plot the component
        fig = _plot_component(slice_2d, title_text, xlabel_text, ylabel_text, alpha)
        figures.append(fig)

    return figures


def _plot_component(
    array: np.ndarray,
    title_text: str,
    xlabel_text: str,
    ylabel_text: str,
    alpha: float = 0.2
) -> Figure:
    """
    Plot a single tensor component as a 2D surface.

    Parameters
    ----------
    array : numpy.ndarray
        2D array containing the tensor component values to plot.

    title_text : str
        Title for the plot, typically in LaTeX format (e.g., "$g_{11}$").

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
