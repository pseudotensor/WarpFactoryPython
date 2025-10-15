"""
Approach 1: Gradual Transition - Benchmark Approach

This approach uses smooth temporal interpolation of the shift vector to
gradually accelerate from rest to constant velocity. This is the "naive"
approach that serves as our baseline for comparison.

Concept:
    β^i(r,t) = v_final × f(r) × S(t)

Where:
- f(r) is the spatial shape function (from Fuchs et al., 2024)
- S(t) is a smooth temporal transition function
- v_final is the target velocity

Hypothesis: Longer transition time τ leads to smaller ∂_t g_μν and thus
smaller energy condition violations. However, violations are not eliminated,
just spread over longer time.

Expected Outcome: 30-50% violation reduction compared to instantaneous transition
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpfactory.core.tensor import Tensor
from warpfactory.metrics.three_plus_one import three_plus_one_builder
from warpfactory.units.constants import c, G
from acceleration_research.time_dependent_framework import (
    TimeDependentMetric,
    transition_sigmoid,
    transition_exponential,
    transition_fermi_dirac,
    transition_polynomial,
    evaluate_energy_conditions_over_time,
    compute_violation_metrics
)


class GradualTransitionWarpDrive:
    """
    Gradual transition warp drive with smooth temporal evolution

    This implements the baseline approach where the shift vector smoothly
    transitions from 0 to v_final over time scale tau.
    """

    def __init__(
        self,
        R1: float = 10.0,  # Inner shell radius (m)
        R2: float = 20.0,  # Outer shell radius (m)
        M: float = 4.49e27,  # Shell mass (kg)
        v_final: float = 0.02,  # Final velocity (fraction of c)
        sigma: float = 0.02,  # Shape function width (m^-1)
        t0: float = 50.0,  # Center time of transition
        tau: float = 25.0,  # Transition time scale
        transition_type: str = "sigmoid"  # Type of transition function
    ):
        """
        Initialize gradual transition warp drive

        Args:
            R1: Inner shell radius in meters
            R2: Outer shell radius in meters
            M: Shell mass in kg
            v_final: Final velocity as fraction of c
            sigma: Shape function width parameter
            t0: Center time of transition (seconds)
            tau: Transition time scale (seconds)
            transition_type: "sigmoid", "exponential", "fermi_dirac", or "polynomial"
        """
        self.R1 = R1
        self.R2 = R2
        self.M = M
        self.v_final = v_final
        self.sigma = sigma
        self.t0 = t0
        self.tau = tau
        self.transition_type = transition_type

        # Derived parameters
        self.R_center = (R1 + R2) / 2.0
        self.Delta_R = (R2 - R1) / 2.0

    def shape_function(self, r: np.ndarray) -> np.ndarray:
        """
        Spatial shape function for shell density and shift vector

        Based on Fuchs et al. (2024) formulation

        Args:
            r: Radial distance from center

        Returns:
            Shape function value between 0 and 1
        """
        # Smooth step centered on shell
        inner_edge = np.tanh(self.sigma * self.Delta_R * (r - self.R1))
        outer_edge = np.tanh(self.sigma * self.Delta_R * (self.R2 - r))

        # Combine: 1 inside shell, 0 outside
        f = 0.25 * (1.0 + inner_edge) * (1.0 + outer_edge)

        return f

    def temporal_transition(self, t: float) -> float:
        """
        Temporal transition function S(t)

        Args:
            t: Time

        Returns:
            Value between 0 (before transition) and 1 (after transition)
        """
        if self.transition_type == "sigmoid":
            return transition_sigmoid(t, self.t0, self.tau)
        elif self.transition_type == "exponential":
            return transition_exponential(t, self.t0 - 2*self.tau, self.tau)
        elif self.transition_type == "fermi_dirac":
            return transition_fermi_dirac(t, self.t0, self.tau, kT=1.0)
        elif self.transition_type == "polynomial":
            return transition_polynomial(t, self.t0 - 2*self.tau, 4*self.tau, n=3)
        else:
            # Linear transition
            if t < self.t0 - self.tau:
                return 0.0
            elif t > self.t0 + self.tau:
                return 1.0
            else:
                return (t - (self.t0 - self.tau)) / (2*self.tau)

    def shell_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        Mass density of shell (constant in time for this approach)

        Args:
            r: Radial distance
            t: Time (unused, shell is static)

        Returns:
            Energy density
        """
        f_spatial = self.shape_function(r)

        # Normalize: integral over shell equals M
        volume_shell = 4.0 * np.pi * self.R_center**2 * self.Delta_R
        rho_0 = self.M / volume_shell

        rho = rho_0 * f_spatial

        return rho

    def shift_vector(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    r: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Time-dependent shift vector

        β^i(r,t) = v_final * c() * f(r) * S(t)

        Args:
            x, y, z: Cartesian coordinates
            r: Radial distance
            t: Time

        Returns:
            (beta_x, beta_y, beta_z) components
        """
        # Spatial shape
        f_spatial = self.shape_function(r)

        # Temporal transition
        S_t = self.temporal_transition(t)

        # Magnitude: v_final * c() * f(r) * S(t)
        v_magnitude = self.v_final * c() * f_spatial * S_t

        # Direction: motion in +x direction
        beta_x = v_magnitude
        beta_y = np.zeros_like(x)
        beta_z = np.zeros_like(x)

        return beta_x, beta_y, beta_z

    def lapse_function(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        Lapse function from shell mass

        Args:
            r: Radial distance
            t: Time (shell is static, so time-independent)

        Returns:
            Lapse function alpha
        """
        # Schwarzschild-like lapse
        r_g = 2.0 * G() * self.M / c()**2

        # Shape function
        f_shell = self.shape_function(r)

        # Lapse
        alpha = np.ones_like(r)
        mask = r > r_g
        alpha[mask] = np.sqrt(1.0 - f_shell[mask] * r_g / r[mask])
        alpha[~mask] = 0.1

        # Ensure minimum value
        alpha = np.maximum(alpha, 0.1)

        return alpha

    def get_metric_3plus1(self, t: float, X: np.ndarray, Y: np.ndarray,
                         Z: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Get 3+1 metric components at time t

        Args:
            t: Time
            X, Y, Z: Spatial coordinate grids

        Returns:
            (alpha, beta_dict, gamma_dict)
        """
        # Radial distance
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R = np.maximum(R, 1e-10)

        # Components
        alpha = self.lapse_function(R, t)
        beta_x, beta_y, beta_z = self.shift_vector(X, Y, Z, R, t)

        beta = {
            0: beta_x,
            1: beta_y,
            2: beta_z
        }

        # Flat spatial metric
        gamma = {}
        for i in range(3):
            for j in range(3):
                if i == j:
                    gamma[(i, j)] = np.ones_like(X)
                else:
                    gamma[(i, j)] = np.zeros_like(X)

        return alpha, beta, gamma

    def get_metric_tensor(self, t: float, X: np.ndarray, Y: np.ndarray,
                         Z: np.ndarray) -> Tensor:
        """
        Get full metric tensor at time t

        Args:
            t: Time
            X, Y, Z: Spatial coordinate grids

        Returns:
            Metric Tensor object
        """
        alpha, beta, gamma = self.get_metric_3plus1(t, X, Y, Z)

        # Build metric
        # Add time dimension to make 4D arrays (required by WarpFactory)
        alpha_4d = alpha[np.newaxis, :, :, :]
        beta_4d = {key: val[np.newaxis, :, :, :] for key, val in beta.items()}
        gamma_4d = {key: val[np.newaxis, :, :, :] for key, val in gamma.items()}

        metric_dict = three_plus_one_builder(alpha_4d, beta_4d, gamma_4d)

        # Create Tensor
        metric = Tensor(
            metric_dict,
            tensor_type="metric",
            name=f"GradualTransition_t{t:.1f}",
            index="covariant",
            coords="cartesian",
            params={
                'R1': self.R1,
                'R2': self.R2,
                'M': self.M,
                'v_final': self.v_final,
                't': t,
                't0': self.t0,
                'tau': self.tau,
                'transition_type': self.transition_type
            }
        )

        return metric


def run_gradual_transition_simulation(
    params: Optional[Dict] = None,
    grid_size: Tuple[int, int, int, int] = (20, 40, 40, 40),
    spatial_extent: float = 100.0,
    verbose: bool = True
) -> Dict:
    """
    Run full simulation of gradual transition approach

    Args:
        params: Dictionary of parameters
        grid_size: [N_t, N_x, N_y, N_z]
        spatial_extent: Size of spatial domain in meters
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    # Default parameters
    if params is None:
        params = {
            'R1': 10.0,
            'R2': 20.0,
            'M': 4.49e27,
            'v_final': 0.02,
            'sigma': 0.02,
            't0': 50.0,
            'tau': 25.0,
            'transition_type': 'sigmoid'
        }

    if verbose:
        print("=" * 70)
        print("APPROACH 1: GRADUAL TRANSITION - Benchmark")
        print("=" * 70)
        print(f"\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        print()

    # Create warp drive
    wd = GradualTransitionWarpDrive(**params)

    # Create time-dependent metric
    time_range = (0.0, params['t0'] + 4*params['tau'])
    spatial_extent_tuple = [
        (-spatial_extent, spatial_extent),
        (-spatial_extent, spatial_extent),
        (-spatial_extent, spatial_extent)
    ]

    def metric_function(t, X, Y, Z):
        return wd.get_metric_tensor(t, X, Y, Z)

    td_metric = TimeDependentMetric(
        grid_size=list(grid_size),
        time_range=time_range,
        spatial_extent=spatial_extent_tuple,
        metric_function=metric_function,
        name="GradualTransition"
    )

    if verbose:
        print("Evaluating energy conditions over time...")
        print(f"Time range: {time_range[0]:.1f} to {time_range[1]:.1f} seconds")
        print(f"Grid size: {grid_size}")
        print()

    # Evaluate energy conditions
    results = evaluate_energy_conditions_over_time(
        td_metric,
        conditions=["Null", "Weak", "Dominant", "Strong"],
        num_angular_vec=50,
        num_time_vec=10,
        verbose=verbose
    )

    # Compute metrics
    if verbose:
        print("\nComputing violation metrics...")

    metrics = compute_violation_metrics(results)

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        for condition, data in metrics.items():
            print(f"\n{condition} Energy Condition:")
            print(f"  Worst violation: {data['worst_violation']:.6e}")
            print(f"  Max magnitude: {data['max_magnitude']:.6e}")
            print(f"  L2 norm: {data['total_violation_L2']:.6e}")
            print(f"  Fraction violating: {data['fraction_violating']:.2%}")
            if data['temporal_extent'] is not None:
                print(f"  Temporal extent: {data['temporal_extent'][0]:.1f} to {data['temporal_extent'][1]:.1f} s")
            if data['peak_time'] is not None:
                print(f"  Peak violation at: {data['peak_time']:.1f} s")

    # Store results
    output = {
        'approach': 'Gradual Transition (Approach 1)',
        'params': params,
        'results': results,
        'metrics': metrics,
        'warp_drive': wd,
        'time_dependent_metric': td_metric
    }

    return output


def compare_transition_types(
    tau_values: list = [10.0, 25.0, 50.0, 100.0],
    transition_types: list = ["sigmoid", "exponential", "polynomial"],
    grid_size: Tuple[int, int, int, int] = (15, 30, 30, 30),
    verbose: bool = True
) -> Dict:
    """
    Compare different transition functions and time scales

    Args:
        tau_values: List of tau values to test
        transition_types: List of transition types
        grid_size: Grid size for simulations
        verbose: Print progress

    Returns:
        Comparison results
    """
    all_results = {}

    for trans_type in transition_types:
        for tau in tau_values:
            key = f"{trans_type}_tau{tau:.0f}"

            if verbose:
                print(f"\n{'='*70}")
                print(f"Testing: {trans_type} with tau = {tau:.0f} seconds")
                print(f"{'='*70}\n")

            params = {
                'R1': 10.0,
                'R2': 20.0,
                'M': 4.49e27,
                'v_final': 0.02,
                'sigma': 0.02,
                't0': 50.0,
                'tau': tau,
                'transition_type': trans_type
            }

            result = run_gradual_transition_simulation(
                params=params,
                grid_size=grid_size,
                spatial_extent=80.0,
                verbose=verbose
            )

            all_results[key] = result

    if verbose:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print("\nNull Energy Condition violations:")
        print(f"{'Configuration':<25} {'Worst':>15} {'L2 Norm':>15} {'Frac Viol':>12}")
        print("-" * 70)

        for key, result in all_results.items():
            nec_data = result['metrics']['Null']
            print(f"{key:<25} {nec_data['worst_violation']:>15.6e} "
                  f"{nec_data['total_violation_L2']:>15.6e} "
                  f"{nec_data['fraction_violating']:>11.2%}")

    return all_results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING APPROACH 1: GRADUAL TRANSITION")
    print("="*70 + "\n")

    # Test with default parameters
    results = run_gradual_transition_simulation(
        grid_size=(10, 20, 20, 20),
        spatial_extent=50.0,
        verbose=True
    )

    print("\n" + "="*70)
    print("Test complete! Results stored in 'results' dictionary.")
    print("="*70)
