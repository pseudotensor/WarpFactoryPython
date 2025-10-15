"""
Tensor operations and utilities

This module provides operations for tensor manipulations including:
- Matrix inversion for 3x3 and 4x4 cell arrays
- Tensor index transformations
- Tensor verification
"""

import numpy as np
import warnings
from typing import Dict, Literal, Optional
from .tensor import Tensor


def get_array_module(arr):
    """Get the appropriate array module (numpy or cupy) for an array"""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp
    except ImportError:
        pass
    return np


def c3_inv(cell_array: Dict[tuple, np.ndarray]) -> Dict[tuple, np.ndarray]:
    """
    Find the inverse of a 3x3 cell array (dictionary)

    Args:
        cell_array: Dictionary with (i,j) keys for i,j in {0,1,2}

    Returns:
        Dictionary: Inverse of the cell array
    """
    # Check dimensions
    if len(cell_array) != 9:
        raise ValueError("Cell array is not 3x3")

    r = cell_array
    xp = get_array_module(r[(0, 0)])

    # Calculate determinant
    det = (r[(0,0)] * r[(1,1)] * r[(2,2)] - r[(0,0)] * r[(1,2)] * r[(2,1)] -
           r[(0,1)] * r[(1,0)] * r[(2,2)] + r[(0,1)] * r[(1,2)] * r[(2,0)] +
           r[(0,2)] * r[(1,0)] * r[(2,1)] - r[(0,2)] * r[(1,1)] * r[(2,0)])

    # Calculate inverse
    inv_cell_array = {
        (0,0): (r[(1,1)] * r[(2,2)] - r[(1,2)] * r[(2,1)]) / det,
        (0,1): (r[(0,2)] * r[(2,1)] - r[(0,1)] * r[(2,2)]) / det,
        (0,2): (r[(0,1)] * r[(1,2)] - r[(0,2)] * r[(1,1)]) / det,
        (1,0): (r[(1,2)] * r[(2,0)] - r[(1,0)] * r[(2,2)]) / det,
        (1,1): (r[(0,0)] * r[(2,2)] - r[(0,2)] * r[(2,0)]) / det,
        (1,2): (r[(0,2)] * r[(1,0)] - r[(0,0)] * r[(1,2)]) / det,
        (2,0): (r[(1,0)] * r[(2,1)] - r[(1,1)] * r[(2,0)]) / det,
        (2,1): (r[(0,1)] * r[(2,0)] - r[(0,0)] * r[(2,1)]) / det,
        (2,2): (r[(0,0)] * r[(1,1)] - r[(0,1)] * r[(1,0)]) / det,
    }

    return inv_cell_array


def c_det(cell_array: Dict[tuple, np.ndarray]) -> np.ndarray:
    """
    Calculate the determinant of a 4x4 cell array

    Args:
        cell_array: Dictionary with (i,j) keys for i,j in {0,1,2,3}

    Returns:
        ndarray: Determinant at each point in spacetime
    """
    r = cell_array

    det = (r[(0,0)] * r[(1,1)] * r[(2,2)] * r[(3,3)] - r[(0,0)] * r[(1,1)] * r[(2,3)] * r[(3,2)] -
           r[(0,0)] * r[(1,2)] * r[(2,1)] * r[(3,3)] + r[(0,0)] * r[(1,2)] * r[(2,3)] * r[(3,1)] +
           r[(0,0)] * r[(1,3)] * r[(2,1)] * r[(3,2)] - r[(0,0)] * r[(1,3)] * r[(2,2)] * r[(3,1)] -
           r[(0,1)] * r[(1,0)] * r[(2,2)] * r[(3,3)] + r[(0,1)] * r[(1,0)] * r[(2,3)] * r[(3,2)] +
           r[(0,1)] * r[(1,2)] * r[(2,0)] * r[(3,3)] - r[(0,1)] * r[(1,2)] * r[(2,3)] * r[(3,0)] -
           r[(0,1)] * r[(1,3)] * r[(2,0)] * r[(3,2)] + r[(0,1)] * r[(1,3)] * r[(2,2)] * r[(3,0)] +
           r[(0,2)] * r[(1,0)] * r[(2,1)] * r[(3,3)] - r[(0,2)] * r[(1,0)] * r[(2,3)] * r[(3,1)] -
           r[(0,2)] * r[(1,1)] * r[(2,0)] * r[(3,3)] + r[(0,2)] * r[(1,1)] * r[(2,3)] * r[(3,0)] +
           r[(0,2)] * r[(1,3)] * r[(2,0)] * r[(3,1)] - r[(0,2)] * r[(1,3)] * r[(2,1)] * r[(3,0)] -
           r[(0,3)] * r[(1,0)] * r[(2,1)] * r[(3,2)] + r[(0,3)] * r[(1,0)] * r[(2,2)] * r[(3,1)] +
           r[(0,3)] * r[(1,1)] * r[(2,0)] * r[(3,2)] - r[(0,3)] * r[(1,1)] * r[(2,2)] * r[(3,0)] -
           r[(0,3)] * r[(1,2)] * r[(2,0)] * r[(3,1)] + r[(0,3)] * r[(1,2)] * r[(2,1)] * r[(3,0)])

    return det


def c4_inv(cell_array: Dict[tuple, np.ndarray]) -> Dict[tuple, np.ndarray]:
    """
    Find the inverse of a 4x4 cell array (dictionary)

    Args:
        cell_array: Dictionary with (i,j) keys for i,j in {0,1,2,3}

    Returns:
        Dictionary: Inverse of the cell array
    """
    # Check dimensions
    if len(cell_array) != 16:
        raise ValueError("Cell array is not 4x4")

    r = cell_array
    det = c_det(r)

    # Calculate inverse using cofactor method
    inv_cell_array = {
        (0,0): (r[(1,1)] * r[(2,2)] * r[(3,3)] - r[(1,1)] * r[(2,3)] * r[(3,2)] -
                r[(1,2)] * r[(2,1)] * r[(3,3)] + r[(1,2)] * r[(2,3)] * r[(3,1)] +
                r[(1,3)] * r[(2,1)] * r[(3,2)] - r[(1,3)] * r[(2,2)] * r[(3,1)]) / det,

        (0,1): (r[(0,1)] * r[(2,3)] * r[(3,2)] - r[(0,1)] * r[(2,2)] * r[(3,3)] +
                r[(0,2)] * r[(2,1)] * r[(3,3)] - r[(0,2)] * r[(2,3)] * r[(3,1)] -
                r[(0,3)] * r[(2,1)] * r[(3,2)] + r[(0,3)] * r[(2,2)] * r[(3,1)]) / det,

        (0,2): (r[(0,1)] * r[(1,2)] * r[(3,3)] - r[(0,1)] * r[(1,3)] * r[(3,2)] -
                r[(0,2)] * r[(1,1)] * r[(3,3)] + r[(0,2)] * r[(1,3)] * r[(3,1)] +
                r[(0,3)] * r[(1,1)] * r[(3,2)] - r[(0,3)] * r[(1,2)] * r[(3,1)]) / det,

        (0,3): (r[(0,1)] * r[(1,3)] * r[(2,2)] - r[(0,1)] * r[(1,2)] * r[(2,3)] +
                r[(0,2)] * r[(1,1)] * r[(2,3)] - r[(0,2)] * r[(1,3)] * r[(2,1)] -
                r[(0,3)] * r[(1,1)] * r[(2,2)] + r[(0,3)] * r[(1,2)] * r[(2,1)]) / det,

        (1,0): (r[(1,0)] * r[(2,3)] * r[(3,2)] - r[(1,0)] * r[(2,2)] * r[(3,3)] +
                r[(1,2)] * r[(2,0)] * r[(3,3)] - r[(1,2)] * r[(2,3)] * r[(3,0)] -
                r[(1,3)] * r[(2,0)] * r[(3,2)] + r[(1,3)] * r[(2,2)] * r[(3,0)]) / det,

        (1,1): (r[(0,0)] * r[(2,2)] * r[(3,3)] - r[(0,0)] * r[(2,3)] * r[(3,2)] -
                r[(0,2)] * r[(2,0)] * r[(3,3)] + r[(0,2)] * r[(2,3)] * r[(3,0)] +
                r[(0,3)] * r[(2,0)] * r[(3,2)] - r[(0,3)] * r[(2,2)] * r[(3,0)]) / det,

        (1,2): (r[(0,0)] * r[(1,3)] * r[(3,2)] - r[(0,0)] * r[(1,2)] * r[(3,3)] +
                r[(0,2)] * r[(1,0)] * r[(3,3)] - r[(0,2)] * r[(1,3)] * r[(3,0)] -
                r[(0,3)] * r[(1,0)] * r[(3,2)] + r[(0,3)] * r[(1,2)] * r[(3,0)]) / det,

        (1,3): (r[(0,0)] * r[(1,2)] * r[(2,3)] - r[(0,0)] * r[(1,3)] * r[(2,2)] -
                r[(0,2)] * r[(1,0)] * r[(2,3)] + r[(0,2)] * r[(1,3)] * r[(2,0)] +
                r[(0,3)] * r[(1,0)] * r[(2,2)] - r[(0,3)] * r[(1,2)] * r[(2,0)]) / det,

        (2,0): (r[(1,0)] * r[(2,1)] * r[(3,3)] - r[(1,0)] * r[(2,3)] * r[(3,1)] -
                r[(1,1)] * r[(2,0)] * r[(3,3)] + r[(1,1)] * r[(2,3)] * r[(3,0)] +
                r[(1,3)] * r[(2,0)] * r[(3,1)] - r[(1,3)] * r[(2,1)] * r[(3,0)]) / det,

        (2,1): (r[(0,0)] * r[(2,3)] * r[(3,1)] - r[(0,0)] * r[(2,1)] * r[(3,3)] +
                r[(0,1)] * r[(2,0)] * r[(3,3)] - r[(0,1)] * r[(2,3)] * r[(3,0)] -
                r[(0,3)] * r[(2,0)] * r[(3,1)] + r[(0,3)] * r[(2,1)] * r[(3,0)]) / det,

        (2,2): (r[(0,0)] * r[(1,1)] * r[(3,3)] - r[(0,0)] * r[(1,3)] * r[(3,1)] -
                r[(0,1)] * r[(1,0)] * r[(3,3)] + r[(0,1)] * r[(1,3)] * r[(3,0)] +
                r[(0,3)] * r[(1,0)] * r[(3,1)] - r[(0,3)] * r[(1,1)] * r[(3,0)]) / det,

        (2,3): (r[(0,0)] * r[(1,3)] * r[(2,1)] - r[(0,0)] * r[(1,1)] * r[(2,3)] +
                r[(0,1)] * r[(1,0)] * r[(2,3)] - r[(0,1)] * r[(1,3)] * r[(2,0)] -
                r[(0,3)] * r[(1,0)] * r[(2,1)] + r[(0,3)] * r[(1,1)] * r[(2,0)]) / det,

        (3,0): (r[(1,0)] * r[(2,2)] * r[(3,1)] - r[(1,0)] * r[(2,1)] * r[(3,2)] +
                r[(1,1)] * r[(2,0)] * r[(3,2)] - r[(1,1)] * r[(2,2)] * r[(3,0)] -
                r[(1,2)] * r[(2,0)] * r[(3,1)] + r[(1,2)] * r[(2,1)] * r[(3,0)]) / det,

        (3,1): (r[(0,0)] * r[(2,1)] * r[(3,2)] - r[(0,0)] * r[(2,2)] * r[(3,1)] -
                r[(0,1)] * r[(2,0)] * r[(3,2)] + r[(0,1)] * r[(2,2)] * r[(3,0)] +
                r[(0,2)] * r[(2,0)] * r[(3,1)] - r[(0,2)] * r[(2,1)] * r[(3,0)]) / det,

        (3,2): (r[(0,0)] * r[(1,2)] * r[(3,1)] - r[(0,0)] * r[(1,1)] * r[(3,2)] +
                r[(0,1)] * r[(1,0)] * r[(3,2)] - r[(0,1)] * r[(1,2)] * r[(3,0)] -
                r[(0,2)] * r[(1,0)] * r[(3,1)] + r[(0,2)] * r[(1,1)] * r[(3,0)]) / det,

        (3,3): (r[(0,0)] * r[(1,1)] * r[(2,2)] - r[(0,0)] * r[(1,2)] * r[(2,1)] -
                r[(0,1)] * r[(1,0)] * r[(2,2)] + r[(0,1)] * r[(1,2)] * r[(2,0)] +
                r[(0,2)] * r[(1,0)] * r[(2,1)] - r[(0,2)] * r[(1,1)] * r[(2,0)]) / det,
    }

    return inv_cell_array


def verify_tensor(input_tensor: Tensor, suppress_msgs: bool = False) -> bool:
    """
    Verify that a tensor struct is properly formatted

    Args:
        input_tensor: Tensor object to verify
        suppress_msgs: If True, suppress informational messages

    Returns:
        bool: True if tensor is valid, False otherwise
    """
    verified = True

    def disp_message(msg: str):
        if not suppress_msgs:
            print(msg)

    # Check if type field exists
    if not hasattr(input_tensor, 'type'):
        warnings.warn('Tensor type does not exist. Must be Either "metric" or "stress-energy"')
        return False

    # Check tensor type
    if input_tensor.type.lower() == "metric":
        disp_message("type: Metric")
    elif input_tensor.type.lower() == "stress-energy":
        disp_message("Type: Stress-Energy")
    else:
        warnings.warn("Unknown type")
        return False

    # Check tensor dictionary
    if not hasattr(input_tensor, 'tensor') or not isinstance(input_tensor.tensor, dict):
        warnings.warn("Tensor is not formatted correctly. Tensor must be a dictionary of 4D arrays.")
        return False

    if len(input_tensor.tensor) != 16:
        warnings.warn("Tensor must have 16 components (4x4)")
        return False

    # Check that all components have same shape and are 4D
    first_shape = input_tensor.tensor[(0, 0)].shape
    if len(first_shape) != 4:
        warnings.warn("Tensor components must be 4D arrays")
        return False

    for i in range(4):
        for j in range(4):
            if (i, j) not in input_tensor.tensor:
                warnings.warn(f"Missing tensor component ({i},{j})")
                return False
            if input_tensor.tensor[(i, j)].shape != first_shape:
                warnings.warn(f"Tensor component ({i},{j}) has inconsistent shape")
                return False

    disp_message("tensor: Verified")

    # Check coords
    if not hasattr(input_tensor, 'coords'):
        warnings.warn("coords: Empty")
        verified = False
    elif input_tensor.coords.lower() == "cartesian":
        disp_message(f"coords: {input_tensor.coords}")
    else:
        warnings.warn("Non-cartesian coordinates are not supported at this time. Set coords to 'cartesian'.")

    # Check index
    if not hasattr(input_tensor, 'index'):
        warnings.warn("index: Empty")
        verified = False
    elif input_tensor.index.lower() in ["contravariant", "covariant", "mixedupdown", "mixeddownup"]:
        disp_message(f"index: {input_tensor.index}")
    else:
        warnings.warn("Unknown index")
        verified = False

    return verified


def change_tensor_index(
    input_tensor: Tensor,
    index: Literal["covariant", "contravariant", "mixedupdown", "mixeddownup"],
    metric_tensor: Optional[Tensor] = None
) -> Tensor:
    """
    Change a tensor's index

    Args:
        input_tensor: Tensor to change the index of
        index: Target index type
        metric_tensor: Metric tensor (required for non-metric tensors)

    Returns:
        Tensor: New tensor with changed index
    """
    # Handle default input arguments
    if metric_tensor is None:
        if input_tensor.type.lower() != "metric":
            raise ValueError("metric_tensor is needed as third input when changing index of non-metric tensors.")
    else:
        if metric_tensor.index.lower() in ["mixedupdown", "mixeddownup"]:
            raise ValueError("Metric tensor cannot be used in mixed index.")

    # Check for valid transformation
    if index.lower() not in ["mixedupdown", "mixeddownup", "covariant", "contravariant"]:
        raise ValueError('Transformation selected is not allowed, use either: "covariant", "contravariant", "mixedupdown", "mixeddownup"')

    # Create output tensor
    output_tensor = input_tensor.copy()

    # Transformations
    if input_tensor.type.lower() == "metric":
        if ((input_tensor.index.lower() == "covariant" and index.lower() == "contravariant") or
            (input_tensor.index.lower() == "contravariant" and index.lower() == "covariant")):
            output_tensor.tensor = c4_inv(input_tensor.tensor)
        elif input_tensor.index.lower() in ["mixedupdown", "mixeddownup"]:
            raise ValueError("Input tensor is a Metric tensor of mixed index.")
        elif index.lower() in ["mixedupdown", "mixeddownup"]:
            raise ValueError("Cannot convert a metric tensor to mixed index.")
    else:
        # Non-metric tensor transformations
        metric = metric_tensor.copy()

        # Contravariant/covariant conversions
        if input_tensor.index.lower() == "covariant" and index.lower() == "contravariant":
            if metric.index.lower() == "covariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "contravariant"
            output_tensor.tensor = _flip_index(input_tensor, metric)

        elif input_tensor.index.lower() == "contravariant" and index.lower() == "covariant":
            if metric.index.lower() == "contravariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "covariant"
            output_tensor.tensor = _flip_index(input_tensor, metric)

        # To mixed conversions
        elif input_tensor.index.lower() == "contravariant" and index.lower() == "mixedupdown":
            if metric.index.lower() == "contravariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "covariant"
            output_tensor.tensor = _mix_index2(input_tensor, metric)

        elif input_tensor.index.lower() == "contravariant" and index.lower() == "mixeddownup":
            if metric.index.lower() == "contravariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "covariant"
            output_tensor.tensor = _mix_index1(input_tensor, metric)

        elif input_tensor.index.lower() == "covariant" and index.lower() == "mixedupdown":
            if metric.index.lower() == "covariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "contravariant"
            output_tensor.tensor = _mix_index1(input_tensor, metric)

        elif input_tensor.index.lower() == "covariant" and index.lower() == "mixeddownup":
            if metric.index.lower() == "covariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "contravariant"
            output_tensor.tensor = _mix_index2(input_tensor, metric)

        # From mixed conversions
        elif input_tensor.index.lower() == "mixedupdown" and index.lower() == "contravariant":
            if metric.index.lower() == "covariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "contravariant"
            output_tensor.tensor = _mix_index2(input_tensor, metric)

        elif input_tensor.index.lower() == "mixedupdown" and index.lower() == "covariant":
            if metric.index.lower() == "contravariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "covariant"
            output_tensor.tensor = _mix_index1(input_tensor, metric)

        elif input_tensor.index.lower() == "mixeddownup" and index.lower() == "covariant":
            if metric.index.lower() == "contravariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "covariant"
            output_tensor.tensor = _mix_index2(input_tensor, metric)

        elif input_tensor.index.lower() == "mixeddownup" and index.lower() == "contravariant":
            if metric.index.lower() == "covariant":
                metric.tensor = c4_inv(metric.tensor)
                metric.index = "contravariant"
            output_tensor.tensor = _mix_index1(input_tensor, metric)

    output_tensor.index = index
    return output_tensor


# Helper functions for index operations
def _flip_index(input_tensor: Tensor, metric_tensor: Tensor) -> Dict[tuple, np.ndarray]:
    """Flip all indices using metric"""
    xp = get_array_module(input_tensor[(0, 0)])
    temp_output = {}
    s = input_tensor.shape

    for i in range(4):
        for j in range(4):
            temp_output[(i, j)] = xp.zeros(s)
            for a in range(4):
                for b in range(4):
                    temp_output[(i, j)] += (input_tensor[(a, b)] *
                                           metric_tensor[(a, i)] *
                                           metric_tensor[(b, j)])
    return temp_output


def _mix_index1(input_tensor: Tensor, metric_tensor: Tensor) -> Dict[tuple, np.ndarray]:
    """Mix first index using metric"""
    xp = get_array_module(input_tensor[(0, 0)])
    temp_output = {}
    s = input_tensor.shape

    for i in range(4):
        for j in range(4):
            temp_output[(i, j)] = xp.zeros(s)
            for a in range(4):
                temp_output[(i, j)] += input_tensor[(a, j)] * metric_tensor[(a, i)]
    return temp_output


def _mix_index2(input_tensor: Tensor, metric_tensor: Tensor) -> Dict[tuple, np.ndarray]:
    """Mix second index using metric"""
    xp = get_array_module(input_tensor[(0, 0)])
    temp_output = {}
    s = input_tensor.shape

    for i in range(4):
        for j in range(4):
            temp_output[(i, j)] = xp.zeros(s)
            for a in range(4):
                temp_output[(i, j)] += input_tensor[(i, a)] * metric_tensor[(a, j)]
    return temp_output
