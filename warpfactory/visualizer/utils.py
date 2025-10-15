"""
Visualization utility functions for WarpFactory.

This module provides helper functions for extracting and visualizing
spacetime tensor data, including custom colormaps and axis labeling.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typing import List, Tuple, Dict, Any, Union


def get_slice_data(plane: List[int], slice_center: List[int], tensor) -> List[Union[range, int]]:
    """
    Extract index data for a 2D slice from 4D tensor data.

    This function creates index selectors for extracting a 2D slice from a 4D
    spacetime tensor by fixing two dimensions at specified values and allowing
    the other two to vary.

    Parameters
    ----------
    plane : list of int
        Two-element list specifying which dimensions to fix (e.g., [0, 1] for t-x plane).
        Values should be in range 0-3 corresponding to [t, x, y, z].
    slice_center : list of int
        Two-element list specifying the index values at which to fix the dimensions.
    tensor : Tensor
        Tensor object with .shape property that returns the 4D dimensions [t, x, y, z].

    Returns
    -------
    list
        Four-element list containing index ranges or fixed indices:
        - For varying dimensions: range object (e.g., range(0, n))
        - For fixed dimensions: int (the slice_center value)

    Examples
    --------
    Extract a slice at constant t=5, x=10 (showing y-z plane):
    >>> index_data = get_slice_data([0, 1], [5, 10], tensor)
    >>> # Returns: [5, 10, range(0, ny), range(0, nz)]

    Extract a slice at constant y=3, z=7 (showing t-x plane):
    >>> index_data = get_slice_data([2, 3], [3, 7], tensor)
    >>> # Returns: [range(0, nt), range(0, nx), 3, 7]

    Notes
    -----
    - The function works with Tensor objects that have a .shape property
    - The plane parameter specifies which dimensions to fix, not which to show
    - The remaining two dimensions will form the 2D slice
    """
    # Get the tensor shape
    s = tensor.shape

    # Initialize with full ranges for all dimensions
    index_data = [range(s[0]), range(s[1]), range(s[2]), range(s[3])]

    # Fix the specified plane dimensions to the slice center values
    index_data[plane[0]] = slice_center[0]
    index_data[plane[1]] = slice_center[1]

    return index_data


def redblue(value: np.ndarray, gradient_num: int = 1024) -> Union[LinearSegmentedColormap, np.ndarray]:
    """
    Create a red-blue diverging colormap centered at zero.

    This function generates a custom colormap that transitions from red (negative)
    through white (zero) to blue (positive), with the white point automatically
    positioned based on the data's min and max values.

    Parameters
    ----------
    value : numpy.ndarray
        Data array used to determine the range and centering of the colormap.
        The colormap will be optimized for this data's min/max values.
    gradient_num : int, optional
        Number of color gradations in the colormap (default: 1024).
        Higher values create smoother gradients.

    Returns
    -------
    numpy.ndarray or matplotlib.colors.Colormap
        RGB colormap array of shape (N, 3) where:
        - Red channel: [R, G, B] values for negative data
        - White: [1, 1, 1] at zero
        - Blue channel: [R, G, B] values for positive data

    Examples
    --------
    Create a colormap for data ranging from -2 to 5:
    >>> data = np.array([[-2, -1, 0], [1, 3, 5]])
    >>> cmap = redblue(data)
    >>> # White will be positioned at 2/7 of the colormap

    Create a colormap for all-positive data (blue only):
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> cmap = redblue(data)
    >>> # Returns blue-to-white gradient

    Create a colormap for all-negative data (red only):
    >>> data = np.array([[-6, -5, -4], [-3, -2, -1]])
    >>> cmap = redblue(data)
    >>> # Returns white-to-red gradient

    Notes
    -----
    - If data contains only zeros, returns pure white [1, 1, 1]
    - If data is all positive, returns blue-to-white gradient
    - If data is all negative, returns white-to-red gradient
    - If data spans zero, creates balanced diverging colormap
    - The white point is positioned proportionally: abs(max)/(abs(max)+abs(min))
    """
    min_value = np.min(value)
    max_value = np.max(value)

    # Case 1: Data doesn't span zero
    if not (min_value <= 0 and max_value >= 0):
        # All positive: blue-to-white gradient
        if min_value > 0 and max_value > 0:
            red = np.linspace(1, 0, round(gradient_num))
            green = np.linspace(1, 0, round(gradient_num))
            blue = np.ones(round(gradient_num))
            return np.column_stack([red, green, blue])

        # All negative: white-to-red gradient
        if min_value < 0 and max_value < 0:
            red = np.ones(round(gradient_num))
            green = np.linspace(0, 1, round(gradient_num))
            blue = np.linspace(0, 1, round(gradient_num))
            return np.column_stack([red, green, blue])

    # Case 2: All zeros
    if min_value == 0 and max_value == 0:
        return np.array([[1, 1, 1]])

    # Case 3: Data spans zero - create diverging colormap
    # Calculate where zero should be positioned in the colormap
    center_val = abs(max_value) / (abs(max_value) + abs(min_value))

    # Red side (negative values): white to red
    num_red = round((1 - center_val) * gradient_num)
    red_side_red = np.ones(num_red)
    red_side_green = np.linspace(0, 1, num_red)
    red_side_blue = np.linspace(0, 1, num_red)
    red_side = np.column_stack([red_side_red, red_side_green, red_side_blue])

    # Blue side (positive values): white to blue
    num_blue = round(center_val * gradient_num)
    blue_side_red = np.linspace(1, 0, num_blue)
    blue_side_green = np.linspace(1, 0, num_blue)
    blue_side_blue = np.ones(num_blue)
    blue_side = np.column_stack([blue_side_red, blue_side_green, blue_side_blue])

    # Combine red and blue sides
    returned_map = np.vstack([red_side, blue_side])

    return returned_map


def label_cartesian_axis(plane: List[int]) -> Tuple[str, str]:
    """
    Generate axis labels for a 2D slice of 4D spacetime.

    Given which two dimensions are fixed (the plane), this function determines
    which two dimensions are shown and returns appropriate axis labels using
    spacetime coordinates (t, x, y, z).

    Parameters
    ----------
    plane : list of int
        Two-element list specifying which dimensions are fixed (hidden).
        Values should be in range 0-3 corresponding to [t, x, y, z].
        For example, [0, 1] means t and x are fixed, so y and z are shown.

    Returns
    -------
    tuple of str
        Two strings: (xlabel, ylabel)
        - xlabel: Label for the horizontal axis
        - ylabel: Label for the vertical axis
        Labels are chosen from ['t', 'x', 'y', 'z']

    Examples
    --------
    Get labels for a slice with t and x fixed (showing y-z plane):
    >>> xlabel, ylabel = label_cartesian_axis([0, 1])
    >>> print(xlabel, ylabel)
    y z

    Get labels for a slice with y and z fixed (showing t-x plane):
    >>> xlabel, ylabel = label_cartesian_axis([2, 3])
    >>> print(xlabel, ylabel)
    t x

    Get labels for a slice with t and y fixed (showing x-z plane):
    >>> xlabel, ylabel = label_cartesian_axis([0, 2])
    >>> print(xlabel, ylabel)
    x z

    Notes
    -----
    - The function returns the labels in sorted order of dimension indices
    - Dimension order is: 0=t (time), 1=x, 2=y, 3=z (space)
    - The plane parameter specifies which dimensions are FIXED, not which are SHOWN
    - The shown dimensions are the complement of the plane dimensions
    """
    labels = ["t", "x", "y", "z"]

    # Find which dimensions are shown (not in plane)
    all_dims = set(range(4))
    plane_dims = set(plane)
    shown_planes = sorted(all_dims - plane_dims)

    xlabel = labels[shown_planes[0]]
    ylabel = labels[shown_planes[1]]

    return xlabel, ylabel


def create_redblue_colormap(name: str = 'redblue') -> LinearSegmentedColormap:
    """
    Create a matplotlib LinearSegmentedColormap version of the redblue colormap.

    This function creates a reusable colormap object that can be registered with
    matplotlib and used with standard plotting functions. Unlike the redblue()
    function which adapts to specific data, this creates a symmetric colormap.

    Parameters
    ----------
    name : str, optional
        Name to register the colormap under (default: 'redblue')

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A diverging colormap from red (-1) through white (0) to blue (+1)

    Examples
    --------
    Create and register the colormap:
    >>> cmap = create_redblue_colormap('my_redblue')
    >>> plt.imshow(data, cmap=cmap)

    Register with matplotlib:
    >>> from matplotlib import colormaps
    >>> cmap = create_redblue_colormap('redblue')
    >>> colormaps.register(cmap)
    >>> plt.imshow(data, cmap='redblue')

    Notes
    -----
    This creates a symmetric colormap suitable for general use. For data-adaptive
    colormaps that position white at the true zero of your data, use redblue() instead.
    """
    colors_dict = {
        'red':   [(0.0, 1.0, 1.0),   # Start with red
                  (0.5, 1.0, 1.0),   # White at center
                  (1.0, 0.0, 0.0)],  # End with no red (blue)

        'green': [(0.0, 0.0, 0.0),   # Start with no green (red)
                  (0.5, 1.0, 1.0),   # White at center
                  (1.0, 0.0, 0.0)],  # End with no green (blue)

        'blue':  [(0.0, 0.0, 0.0),   # Start with no blue (red)
                  (0.5, 1.0, 1.0),   # White at center
                  (1.0, 1.0, 1.0)]   # End with blue
    }

    return LinearSegmentedColormap(name, colors_dict)
