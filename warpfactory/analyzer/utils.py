"""
Utility functions for analyzer module
"""

import numpy as np
from typing import Dict, Literal
from ..core.tensor import Tensor
from ..core.tensor_ops import c4_inv, get_array_module, verify_tensor


def get_even_points_on_sphere(R: float, num_points: int, use_gpu: bool = False) -> np.ndarray:
    """
    Generate evenly distributed points on a sphere using golden ratio spiral

    Args:
        R: Radius of sphere
        num_points: Number of points to generate
        use_gpu: Whether to use GPU arrays

    Returns:
        ndarray: 3xN array of points on sphere [x, y, z]
    """
    xp = np

    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            pass

    golden_ratio = (1 + xp.sqrt(5)) / 2

    if use_gpu and xp != np:
        Vector = xp.zeros((3, num_points))
    else:
        Vector = np.zeros((3, num_points))

    for i in range(num_points):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / num_points)

        Vector[0, i] = R * np.cos(theta) * np.sin(phi)
        Vector[1, i] = R * np.sin(theta) * np.sin(phi)
        Vector[2, i] = R * np.cos(phi)

    return np.real(Vector) if xp == np else xp.real(Vector)


def generate_uniform_field(
    field_type: Literal["nulllike", "timelike"],
    num_angular_vec: int,
    num_time_vec: int = 10,
    try_gpu: bool = False
) -> np.ndarray:
    """
    Generate a uniform vector field for energy condition testing

    Args:
        field_type: Type of vectors ("nulllike" or "timelike")
        num_angular_vec: Number of evenly spaced spatial vectors
        num_time_vec: Number of temporal shells for timelike vectors
        try_gpu: Whether to use GPU arrays

    Returns:
        ndarray: Vector field array
    """
    xp = np
    if try_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            pass

    if field_type.lower() not in ["nulllike", "timelike"]:
        raise ValueError('Vector field type not recognized, use either: "nulllike", "timelike"')

    if field_type.lower() == "timelike":
        # Generate timelike vectors: c²t² > r²
        bb = np.linspace(0, 1, num_time_vec)
        VecField = np.ones((4, num_angular_vec, num_time_vec))

        for jj in range(num_time_vec):
            # Build vector field in cartesian coordinates
            spatial_points = get_even_points_on_sphere(1 - bb[jj], num_angular_vec, False)
            VecField[0, :, jj] = 1.0
            VecField[1:4, :, jj] = spatial_points

            # Normalize
            norm = np.sqrt(
                VecField[0, :, jj]**2 + VecField[1, :, jj]**2 +
                VecField[2, :, jj]**2 + VecField[3, :, jj]**2
            )
            VecField[:, :, jj] = VecField[:, :, jj] / norm

    elif field_type.lower() == "nulllike":
        # Build vector field in cartesian coordinates
        spatial_points = get_even_points_on_sphere(1, num_angular_vec, False)
        VecField = np.ones((4, num_angular_vec))
        VecField[1:4, :] = spatial_points

        # Normalize
        norm = np.sqrt(
            VecField[0, :]**2 + VecField[1, :]**2 +
            VecField[2, :]**2 + VecField[3, :]**2
        )
        VecField = VecField / norm

    # Convert to GPU if requested
    if try_gpu and xp != np:
        VecField = xp.asarray(VecField)

    return VecField


def get_inner_product(
    vec_a: Dict,
    vec_b: Dict,
    metric: Tensor
) -> np.ndarray:
    """
    Take the inner product of two vector fields with a metric

    Args:
        vec_a: First vector (dict with 'field' and 'index' keys)
        vec_b: Second vector (dict with 'field' and 'index' keys)
        metric: Metric tensor

    Returns:
        ndarray: Inner product at each spacetime point
    """
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    s = metric.shape
    xp = get_array_module(metric[(0, 0)])
    innerprod = xp.zeros(s) if xp == np else xp.zeros(s, dtype=metric[(0, 0)].dtype)

    # Check if indices are different (one up, one down)
    if vec_a['index'].lower() != vec_b['index'].lower():
        # Direct contraction
        for mu in range(4):
            for nu in range(4):
                innerprod = innerprod + vec_a['field'][mu] * vec_b['field'][nu]
    else:
        # Need metric to contract
        metric_dict = metric.tensor

        if vec_a['index'].lower() == metric.index.lower():
            metric_dict = c4_inv(metric_dict)

        for mu in range(4):
            for nu in range(4):
                innerprod = innerprod + vec_a['field'][mu] * vec_b['field'][nu] * metric_dict[(mu, nu)]

    return innerprod


def get_trace(tensor: Tensor, metric: Tensor) -> np.ndarray:
    """
    Calculate the trace of a tensor

    Args:
        tensor: Tensor to trace
        metric: Metric tensor

    Returns:
        ndarray: Trace at each spacetime point
    """
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    xp = get_array_module(metric[(0, 0)])
    s = metric.shape
    Trace = xp.zeros(s) if xp == np else xp.zeros(s, dtype=metric[(0, 0)].dtype)

    metric_dict = metric.tensor

    if tensor.index.lower() == metric.index.lower():
        metric_dict = c4_inv(metric_dict)

    for a in range(4):
        for b in range(4):
            Trace = Trace + metric_dict[(a, b)] * tensor[(a, b)]

    return Trace
