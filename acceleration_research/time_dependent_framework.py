"""
Time-Dependent Warp Drive Framework

This module extends WarpFactory to handle time-dependent metrics needed for
acceleration research. It provides:
- TimeDependentMetric class for metrics that vary in time
- Time derivative computation using finite differences
- Energy condition evaluation over time
- Temporal transition functions (sigmoid, exponential, Fermi-Dirac)
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import sys
import os

# Add parent directory to path to import warpfactory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpfactory.core.tensor import Tensor
from warpfactory.metrics.three_plus_one import three_plus_one_builder, three_plus_one_decomposer
from warpfactory.analyzer.energy_conditions import get_energy_conditions
from warpfactory.solver.einstein import calculate_einstein_tensor
from warpfactory.core.tensor_ops import change_tensor_index


class TimeDependentMetric:
    """
    Time-dependent spacetime metric for warp drive acceleration

    This class manages metrics that vary in time, computing time derivatives
    and evaluating energy conditions at each time slice.

    Attributes:
        grid_size: Spacetime grid dimensions [N_t, N_x, N_y, N_z]
        time_range: [t_start, t_end] in seconds
        spatial_extent: [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters
        metric_function: Function(t, x, y, z) -> Tensor that returns metric at time t
        name: Descriptive name for this metric
    """

    def __init__(
        self,
        grid_size: List[int],
        time_range: Tuple[float, float],
        spatial_extent: List[Tuple[float, float]],
        metric_function: Callable,
        name: str = "TimeDependentMetric"
    ):
        """
        Initialize time-dependent metric

        Args:
            grid_size: [N_t, N_x, N_y, N_z]
            time_range: (t_start, t_end) in seconds
            spatial_extent: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            metric_function: Function that returns metric at time t
            name: Name identifier
        """
        self.grid_size = grid_size
        self.time_range = time_range
        self.spatial_extent = spatial_extent
        self.metric_function = metric_function
        self.name = name

        # Create spacetime grid
        self.t_array = np.linspace(time_range[0], time_range[1], grid_size[0])
        self.x_array = np.linspace(spatial_extent[0][0], spatial_extent[0][1], grid_size[1])
        self.y_array = np.linspace(spatial_extent[1][0], spatial_extent[1][1], grid_size[2])
        self.z_array = np.linspace(spatial_extent[2][0], spatial_extent[2][1], grid_size[3])

        self.dt = self.t_array[1] - self.t_array[0] if len(self.t_array) > 1 else 1.0
        self.dx = self.x_array[1] - self.x_array[0] if len(self.x_array) > 1 else 1.0
        self.dy = self.y_array[1] - self.y_array[0] if len(self.y_array) > 1 else 1.0
        self.dz = self.z_array[1] - self.z_array[0] if len(self.z_array) > 1 else 1.0

        # Storage for metrics at each time slice
        self.metrics = {}

    def get_metric_at_time(self, t_index: int) -> Tensor:
        """
        Get metric tensor at specific time index

        Args:
            t_index: Time index in grid

        Returns:
            Tensor: Metric at that time
        """
        if t_index in self.metrics:
            return self.metrics[t_index]

        # Compute metric at this time
        t = self.t_array[t_index]
        X, Y, Z = np.meshgrid(self.x_array, self.y_array, self.z_array, indexing='ij')

        metric = self.metric_function(t, X, Y, Z)
        self.metrics[t_index] = metric

        return metric

    def compute_time_derivative(self, t_index: int, order: int = 2) -> Tensor:
        """
        Compute time derivative of metric at given time index

        Uses finite difference:
        - order=2: (g(t+dt) - g(t-dt)) / (2*dt)
        - order=4: (-g(t+2dt) + 8g(t+dt) - 8g(t-dt) + g(t-2dt)) / (12*dt)

        Args:
            t_index: Time index
            order: Finite difference order (2 or 4)

        Returns:
            Tensor: Time derivative dg/dt at this time
        """
        if order == 2:
            if t_index == 0:
                # Forward difference at start
                g_0 = self.get_metric_at_time(t_index)
                g_1 = self.get_metric_at_time(t_index + 1)
                dg_dt_dict = {}
                for key in g_0.tensor.keys():
                    dg_dt_dict[key] = (g_1[key] - g_0[key]) / self.dt
            elif t_index == len(self.t_array) - 1:
                # Backward difference at end
                g_0 = self.get_metric_at_time(t_index)
                g_m1 = self.get_metric_at_time(t_index - 1)
                dg_dt_dict = {}
                for key in g_0.tensor.keys():
                    dg_dt_dict[key] = (g_0[key] - g_m1[key]) / self.dt
            else:
                # Central difference
                g_p1 = self.get_metric_at_time(t_index + 1)
                g_m1 = self.get_metric_at_time(t_index - 1)
                dg_dt_dict = {}
                for key in g_p1.tensor.keys():
                    dg_dt_dict[key] = (g_p1[key] - g_m1[key]) / (2.0 * self.dt)

        elif order == 4:
            if t_index < 2 or t_index >= len(self.t_array) - 2:
                # Fall back to 2nd order at boundaries
                return self.compute_time_derivative(t_index, order=2)
            else:
                # 4th order central difference
                g_p2 = self.get_metric_at_time(t_index + 2)
                g_p1 = self.get_metric_at_time(t_index + 1)
                g_m1 = self.get_metric_at_time(t_index - 1)
                g_m2 = self.get_metric_at_time(t_index - 2)
                dg_dt_dict = {}
                for key in g_p1.tensor.keys():
                    dg_dt_dict[key] = (
                        -g_p2[key] + 8.0*g_p1[key] - 8.0*g_m1[key] + g_m2[key]
                    ) / (12.0 * self.dt)
        else:
            raise ValueError("Order must be 2 or 4")

        # Create derivative tensor
        g_0 = self.get_metric_at_time(t_index)
        dg_dt = Tensor(
            dg_dt_dict,
            tensor_type="metric",
            name=f"{self.name}_dg_dt",
            index="covariant",
            coords="cartesian",
            params=g_0.params
        )

        return dg_dt


def transition_sigmoid(t: float, t0: float, tau: float) -> float:
    """
    Sigmoid transition function S(t) = 0.5 * (1 + tanh((t - t0) / tau))

    Smoothly transitions from 0 to 1 around t = t0 with width tau

    Args:
        t: Current time
        t0: Center time of transition
        tau: Width of transition (larger = slower)

    Returns:
        Value between 0 and 1
    """
    return 0.5 * (1.0 + np.tanh((t - t0) / tau))


def transition_exponential(t: float, t0: float, tau: float) -> float:
    """
    Exponential transition S(t) = 1 - exp(-(t - t0) / tau) for t > t0, else 0

    Args:
        t: Current time
        t0: Start time
        tau: Time constant

    Returns:
        Value between 0 and 1
    """
    if np.any(t < t0):
        result = np.zeros_like(t)
        mask = t >= t0
        result[mask] = 1.0 - np.exp(-(t[mask] - t0) / tau)
        return result
    else:
        return 1.0 - np.exp(-(t - t0) / tau)


def transition_fermi_dirac(t: float, t0: float, tau: float, kT: float = 1.0) -> float:
    """
    Fermi-Dirac transition S(t) = 1 / (1 + exp(-(t - t0) / (kT * tau)))

    Args:
        t: Current time
        t0: Center time
        tau: Width
        kT: Temperature parameter (default 1.0)

    Returns:
        Value between 0 and 1
    """
    return 1.0 / (1.0 + np.exp(-(t - t0) / (kT * tau)))


def transition_polynomial(t: float, t0: float, tau: float, n: int = 3) -> float:
    """
    Polynomial transition with continuous derivatives up to order n

    Uses s = (t - t0) / tau, then S(s) = 0 for s < 0, 1 for s > 1,
    and smooth polynomial in between

    Args:
        t: Current time
        t0: Start time
        tau: Duration
        n: Polynomial order (higher = smoother)

    Returns:
        Value between 0 and 1
    """
    s = (t - t0) / tau
    if np.any(s < 0):
        result = np.zeros_like(s)
        mask1 = (s >= 0) & (s <= 1)
        mask2 = s > 1
        if n == 3:
            # Cubic: 3s^2 - 2s^3
            result[mask1] = 3*s[mask1]**2 - 2*s[mask1]**3
        elif n == 5:
            # Quintic: 10s^3 - 15s^4 + 6s^5
            result[mask1] = 10*s[mask1]**3 - 15*s[mask1]**4 + 6*s[mask1]**5
        result[mask2] = 1.0
        return result
    else:
        if np.all(s > 1):
            return np.ones_like(s)
        elif n == 3:
            return 3*s**2 - 2*s**3
        elif n == 5:
            return 10*s**3 - 15*s**4 + 6*s**5
        else:
            raise ValueError("n must be 3 or 5")


def evaluate_energy_conditions_over_time(
    time_dependent_metric: TimeDependentMetric,
    conditions: List[str] = ["Null", "Weak", "Dominant", "Strong"],
    num_angular_vec: int = 50,
    num_time_vec: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Evaluate energy conditions at all time slices

    Args:
        time_dependent_metric: TimeDependentMetric object
        conditions: List of conditions to check
        num_angular_vec: Number of spatial vectors to sample
        num_time_vec: Number of temporal vectors to sample
        verbose: Print progress

    Returns:
        Dictionary with results for each condition and time
    """
    results = {
        'times': time_dependent_metric.t_array,
        'conditions': {}
    }

    for condition in conditions:
        if verbose:
            print(f"Evaluating {condition} Energy Condition...")

        violations_over_time = []
        max_violations = []
        min_violations = []
        mean_violations = []

        for t_idx in range(len(time_dependent_metric.t_array)):
            if verbose and t_idx % max(1, len(time_dependent_metric.t_array) // 10) == 0:
                print(f"  Time step {t_idx+1}/{len(time_dependent_metric.t_array)}")

            # Get metric at this time
            metric = time_dependent_metric.get_metric_at_time(t_idx)

            # Compute stress-energy tensor from metric
            try:
                from warpfactory.solver.energy import get_energy_tensor
                stress_energy = get_energy_tensor(
                    metric,
                    try_gpu=False,
                    diff_order='second'
                )

                # Evaluate energy condition
                violation_map, _, _ = get_energy_conditions(
                    stress_energy,
                    metric,
                    condition,
                    num_angular_vec=num_angular_vec,
                    num_time_vec=num_time_vec,
                    return_vec=False,
                    try_gpu=False
                )

                # Store statistics (remove spatial dimensions, keep single time slice)
                # violation_map has shape [N_t, N_x, N_y, N_z], take slice at t_idx
                if len(violation_map.shape) == 4:
                    violation_slice = violation_map[0, :, :, :]  # Single time slice
                else:
                    violation_slice = violation_map

                violations_over_time.append(violation_slice)
                max_violations.append(np.nanmax(violation_slice))
                min_violations.append(np.nanmin(violation_slice))
                mean_violations.append(np.nanmean(violation_slice))

            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed at time {t_idx}: {e}")
                violations_over_time.append(None)
                max_violations.append(np.nan)
                min_violations.append(np.nan)
                mean_violations.append(np.nan)

        results['conditions'][condition] = {
            'violations_over_time': violations_over_time,
            'max_violations': np.array(max_violations),
            'min_violations': np.array(min_violations),
            'mean_violations': np.array(mean_violations)
        }

    return results


def compute_violation_metrics(results: Dict) -> Dict:
    """
    Compute quantitative metrics from energy condition results

    Args:
        results: Output from evaluate_energy_conditions_over_time

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    for condition, data in results['conditions'].items():
        max_viol = data['max_violations']
        min_viol = data['min_violations']
        mean_viol = data['mean_violations']

        # Overall statistics
        metrics[condition] = {
            'worst_violation': np.nanmin(min_viol),  # Most negative value
            'max_magnitude': np.nanmax(np.abs(min_viol)),
            'total_violation_L2': np.sqrt(np.nansum(min_viol[min_viol < 0]**2)),
            'fraction_violating': np.sum(min_viol < 0) / len(min_viol),
            'temporal_extent': None,
            'peak_time': None
        }

        # Temporal extent (when violations occur)
        violating_indices = np.where(min_viol < 0)[0]
        if len(violating_indices) > 0:
            t_start = results['times'][violating_indices[0]]
            t_end = results['times'][violating_indices[-1]]
            metrics[condition]['temporal_extent'] = (t_start, t_end)

            # Time of peak violation
            worst_idx = np.nanargmin(min_viol)
            metrics[condition]['peak_time'] = results['times'][worst_idx]

    return metrics


def compare_approaches(results_dict: Dict[str, Dict]) -> Dict:
    """
    Compare multiple approaches quantitatively

    Args:
        results_dict: Dictionary mapping approach names to their results

    Returns:
        Comparison metrics
    """
    comparison = {
        'approaches': list(results_dict.keys()),
        'comparison': {}
    }

    # For each energy condition, compare across approaches
    first_result = list(results_dict.values())[0]
    conditions = list(first_result['conditions'].keys())

    for condition in conditions:
        comparison['comparison'][condition] = {}

        for approach_name, results in results_dict.items():
            metrics = compute_violation_metrics(results)
            comparison['comparison'][condition][approach_name] = metrics[condition]

    # Rank approaches by performance
    for condition in conditions:
        rankings = []
        for approach in comparison['approaches']:
            worst = comparison['comparison'][condition][approach]['worst_violation']
            rankings.append((approach, worst))

        # Sort by worst violation (less negative is better)
        rankings.sort(key=lambda x: x[1], reverse=True)
        comparison['comparison'][condition]['rankings'] = rankings

    return comparison


if __name__ == "__main__":
    print("Time-Dependent Framework Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("- TimeDependentMetric class")
    print("- Time derivative computation")
    print("- Energy condition evaluation over time")
    print("- Transition functions (sigmoid, exponential, Fermi-Dirac, polynomial)")
    print("- Violation metrics and comparison tools")
    print("\nImport this module to use in acceleration research approaches.")
