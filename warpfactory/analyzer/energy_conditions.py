"""
Energy condition evaluations

The energy conditions constrain the stress-energy tensor to be physically
reasonable. This module evaluates:
- Null Energy Condition (NEC)
- Weak Energy Condition (WEC)
- Dominant Energy Condition (DEC)
- Strong Energy Condition (SEC)
"""

import numpy as np
import warnings
from typing import Dict, Literal, Optional, Tuple
from ..core.tensor import Tensor
from ..core.tensor_ops import change_tensor_index, verify_tensor, get_array_module
from ..metrics.minkowski import get_minkowski_metric
from .utils import generate_uniform_field, get_inner_product, get_trace


def get_energy_conditions(
    energy_tensor: Tensor,
    metric: Tensor,
    condition: Literal["Null", "Weak", "Dominant", "Strong"],
    num_angular_vec: int = 100,
    num_time_vec: int = 10,
    return_vec: bool = False,
    try_gpu: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get the energy conditions of an energy tensor

    Args:
        energy_tensor: Stress-energy tensor
        metric: Metric tensor
        condition: Which energy condition to evaluate
        num_angular_vec: Number of equally spaced spatial vectors to evaluate
        num_time_vec: Number of equally spaced temporal shells to evaluate
        return_vec: Whether to return all evaluations and their vectors
        try_gpu: Whether to use GPU computation

    Returns:
        Tuple of (map, vec, vector_field_out):
            - map: Most violating evaluation at every point in spacetime
            - vec: Evaluations for every vector (if return_vec=True, else None)
            - vector_field_out: Vector field used (if return_vec=True, else None)
    """
    # Check correct condition input
    if condition not in ["Null", "Weak", "Dominant", "Strong"]:
        raise ValueError('Incorrect energy condition input, use either: "Null", "Weak", "Dominant", "Strong"')

    # Warning for non-cartesian coordinates
    if metric.coords.lower() != 'cartesian':
        warnings.warn('Evaluation not verified for coordinate systems other than Cartesian!')

    # Check tensor formats are correct
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")
    if not verify_tensor(energy_tensor, suppress_msgs=True):
        raise ValueError("Stress-energy is not verified. Please verify stress-energy using verify_tensor(energy_tensor).")

    # Convert arrays to GPU if needed
    if try_gpu:
        try:
            import cupy as cp
            energy_tensor = energy_tensor.to_gpu()
            metric = metric.to_gpu()
        except ImportError:
            warnings.warn("CuPy not installed, using CPU")
            try_gpu = False

    # Get size of spacetime
    a, b, c, d = metric.shape
    xp = get_array_module(metric[(0, 0)])

    # Build vector fields
    if condition in ["Null", "Dominant"]:
        field_type = "nulllike"
    elif condition in ["Weak", "Strong"]:
        field_type = "timelike"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    vec_field = generate_uniform_field(field_type, num_angular_vec, num_time_vec, try_gpu)

    # Declare variables
    if try_gpu and xp != np:
        vec_field = xp.asarray(vec_field)
        map_result = xp.full((a, b, c, d), xp.nan)
        if return_vec:
            if condition in ["Null", "Dominant"]:
                vec_result = xp.zeros((a, b, c, d, num_angular_vec))
            else:
                vec_result = xp.zeros((a, b, c, d, num_angular_vec, num_time_vec))
    else:
        map_result = np.full((a, b, c, d), np.nan)
        if return_vec:
            if condition in ["Null", "Dominant"]:
                vec_result = np.zeros((a, b, c, d, num_angular_vec))
            else:
                vec_result = np.zeros((a, b, c, d, num_angular_vec, num_time_vec))

    # Null Energy Condition
    if condition == "Null":
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for ii in range(num_angular_vec):
            temp = xp.zeros((a, b, c, d)) if xp == np else xp.zeros((a, b, c, d), dtype=metric[(0, 0)].dtype)
            for mu in range(4):
                for nu in range(4):
                    temp = temp + energy_tensor[(mu, nu)] * vec_field[mu, ii] * vec_field[nu, ii]

            map_result = xp.minimum(map_result, temp) if xp != np else np.fmin(map_result, temp)
            if return_vec:
                vec_result[:, :, :, :, ii] = temp

    # Weak Energy Condition
    elif condition == "Weak":
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = xp.zeros((a, b, c, d)) if xp == np else xp.zeros((a, b, c, d), dtype=metric[(0, 0)].dtype)

                for mu in range(4):
                    for nu in range(4):
                        temp = temp + energy_tensor[(mu, nu)] * vec_field[mu, ii, jj] * vec_field[nu, ii, jj]

                map_result = xp.minimum(map_result, temp) if xp != np else np.fmin(map_result, temp)
                if return_vec:
                    vec_result[:, :, :, :, ii, jj] = temp

    # Dominant Energy Condition
    elif condition == "Dominant":
        # Build Minkowski reference metric
        metric_minkowski = get_minkowski_metric([a, b, c, d])
        if try_gpu:
            metric_minkowski = metric_minkowski.to_gpu()
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        # Convert energy tensor to mixed index
        energy_tensor = change_tensor_index(energy_tensor, "mixedupdown", metric_minkowski)

        for ii in range(num_angular_vec):
            temp = xp.zeros((a, b, c, d, 4)) if xp == np else xp.zeros((a, b, c, d, 4), dtype=metric[(0, 0)].dtype)

            for mu in range(4):
                for nu in range(4):
                    temp[:, :, :, :, mu] = temp[:, :, :, :, mu] - energy_tensor[(mu, nu)] * vec_field[nu, ii]

            # Create vector struct
            vector_dict = {
                'field': [temp[:, :, :, :, i] for i in range(4)],
                'index': "contravariant",
                'type': "4-vector"
            }

            # Find inner product to determine if timelike or null
            diff = get_inner_product(vector_dict, vector_dict, metric_minkowski)
            diff = xp.sign(diff) * xp.sqrt(xp.abs(diff))

            map_result = xp.maximum(map_result, diff) if xp != np else np.fmax(map_result, diff)
            if return_vec:
                vec_result[:, :, :, :, ii] = diff

        # Flip sign for consistency with other conditions
        map_result = -map_result
        if return_vec:
            vec_result = -vec_result

    # Strong Energy Condition
    elif condition == "Strong":
        # Build Minkowski reference metric
        metric_minkowski = get_minkowski_metric([a, b, c, d])
        if try_gpu:
            metric_minkowski = metric_minkowski.to_gpu()
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        # Make sure energy tensor is covariant
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric_minkowski)

        # Find the trace
        E_trace = get_trace(energy_tensor, metric_minkowski)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = xp.zeros((a, b, c, d)) if xp == np else xp.zeros((a, b, c, d), dtype=metric[(0, 0)].dtype)

                for mu in range(4):
                    for nu in range(4):
                        temp = temp + (
                            (energy_tensor[(mu, nu)] - 0.5 * E_trace * metric_minkowski[(mu, nu)]) *
                            vec_field[mu, ii, jj] * vec_field[nu, ii, jj]
                        )

                map_result = xp.minimum(map_result, temp) if xp != np else np.fmin(map_result, temp)
                if return_vec:
                    vec_result[:, :, :, :, ii, jj] = temp

    # Convert from GPU if needed
    if try_gpu and xp != np:
        map_result = xp.asnumpy(map_result)
        if return_vec:
            vec_result = xp.asnumpy(vec_result)

    # Return results
    if return_vec:
        vector_field_out = vec_field
        return map_result, vec_result, vector_field_out
    else:
        return map_result, None, None
