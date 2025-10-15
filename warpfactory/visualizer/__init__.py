"""
WarpFactory Visualizer Module

This module provides visualization utilities for WarpFactory spacetime metrics
and tensor fields using matplotlib.
"""

from .utils import get_slice_data, redblue, label_cartesian_axis
from .plot_tensor import plot_tensor
from .plot_three_plus_one import plot_three_plus_one

__all__ = [
    'get_slice_data',
    'redblue',
    'label_cartesian_axis',
    'plot_tensor',
    'plot_three_plus_one',
]
