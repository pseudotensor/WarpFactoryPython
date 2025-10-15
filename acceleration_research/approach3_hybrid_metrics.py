"""
Approach 3: Hybrid Metrics - Staged Acceleration
HIGHEST PRIORITY APPROACH

This approach separates warp drive acceleration into distinct physical stages:
1. Stage 1 (Shell Formation): Create stationary matter shell with M_ADM > 0
2. Stage 2 (Shift Addition): Add shift vector to existing shell, β increases
3. Stage 3 (Coasting): Reach constant velocity, transition to solved case

Key Hypothesis: Pre-existing positive ADM mass provides "energy budget" that
helps satisfy energy conditions during shift vector spin-up.

Physical Motivation:
- Shell formation (Stage 1) is just standard GR - known to be physical
- Constant velocity warp (Stage 3) is proven physical (Fuchs et al., 2024)
- Question: Is the transition between them (Stage 2) also physical?
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
    transition_polynomial,
    evaluate_energy_conditions_over_time,
    compute_violation_metrics
)


class HybridMetricWarpDrive:
    """
    Hybrid metric warp drive with staged acceleration

    This class implements the staged acceleration approach where shell formation
    and shift vector addition are separated temporally.
    """

    def __init__(
        self,
        R1: float = 10.0,  # Inner shell radius (m)
        R2: float = 20.0,  # Outer shell radius (m)
        M: float = 4.49e27,  # Shell mass (kg), default = 2.365 Jupiter masses
        v_final: float = 0.02,  # Final velocity (fraction of c)
        sigma: float = 0.02,  # Shape function width (m^-1)
        t1: float = 0.0,  # Stage 1 end time (shell formation complete)
        t2: float = 50.0,  # Stage 2 start time (shift spin-up begins)
        t3: float = 150.0,  # Stage 3 start time (constant velocity reached)
        transition_type: str = "sigmoid",  # Transition function type
        stage_mode: str = "sequential"  # "sequential" or "overlapping"
    ):
        """
        Initialize hybrid metric warp drive

        Args:
            R1: Inner shell radius in meters
            R2: Outer shell radius in meters
            M: Shell mass in kg
            v_final: Final velocity as fraction of c
            sigma: Shape function width parameter
            t1: Time when shell formation completes
            t2: Time when shift spin-up begins
            t3: Time when constant velocity reached
            transition_type: "sigmoid", "exponential", or "polynomial"
            stage_mode: "sequential" (separated) or "overlapping" (simultaneous)
        """
        self.R1 = R1
        self.R2 = R2
        self.M = M
        self.v_final = v_final
        self.sigma = sigma
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.transition_type = transition_type
        self.stage_mode = stage_mode

        # Derived parameters
        self.R_center = (R1 + R2) / 2.0
        self.Delta_R = (R2 - R1) / 2.0

        # Transition time constants
        self.tau_shell = (t1 - 0) / 4.0  # Shell formation time constant
        self.tau_shift = (t3 - t2) / 4.0  # Shift spin-up time constant

    def shape_function(self, r: np.ndarray) -> np.ndarray:
        """
        Spatial shape function for shell density and shift vector

        Uses smooth step function centered on shell

        Args:
            r: Radial distance from center

        Returns:
            Shape function value between 0 and 1
        """
        # Smooth step from 0 (at R1) to 1 (at R_center) to 0 (at R2)
        x = (r - self.R_center) / self.Delta_R

        # Use tanh-based smooth function
        inner_edge = np.tanh(self.sigma * self.Delta_R * (r - self.R1))
        outer_edge = np.tanh(self.sigma * self.Delta_R * (self.R2 - r))

        # Combine: 1 inside shell, 0 outside
        f = 0.25 * (1.0 + inner_edge) * (1.0 + outer_edge)

        return f

    def shell_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        Mass density of shell as function of position and time

        Args:
            r: Radial distance
            t: Time

        Returns:
            Energy density in geometric units
        """
        # Stage 1: Shell formation (0 to t1)
        if self.stage_mode == "sequential":
            if t < self.t1:
                # Shell growing
                if self.transition_type == "sigmoid":
                    S_shell = transition_sigmoid(t, self.t1/2, self.tau_shell)
                elif self.transition_type == "exponential":
                    S_shell = transition_exponential(t, 0, self.tau_shell)
                elif self.transition_type == "polynomial":
                    S_shell = transition_polynomial(t, 0, self.t1, n=3)
                else:
                    S_shell = np.clip(t / self.t1, 0, 1)
            else:
                # Shell fully formed
                S_shell = 1.0
        else:  # overlapping mode
            # Shell grows throughout acceleration
            if t < self.t3:
                if self.transition_type == "sigmoid":
                    S_shell = transition_sigmoid(t, self.t3/2, self.t3/4)
                else:
                    S_shell = np.clip(t / self.t3, 0, 1)
            else:
                S_shell = 1.0

        # Spatial distribution
        f_spatial = self.shape_function(r)

        # Total density
        # ρ = M / (4π R_center^2 Delta_R) * f(r) * S(t)
        # Normalize so that integral over shell equals M
        volume_shell = 4.0 * np.pi * self.R_center**2 * self.Delta_R
        rho_0 = self.M / volume_shell

        rho = rho_0 * f_spatial * S_shell

        return rho

    def shift_vector(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    r: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shift vector components as function of position and time

        Args:
            x, y, z: Cartesian coordinates
            r: Radial distance
            t: Time

        Returns:
            (beta_x, beta_y, beta_z) components
        """
        # Stage 2: Shift spin-up (t2 to t3)
        if self.stage_mode == "sequential":
            if t < self.t2:
                # No shift yet
                S_shift = 0.0
            elif t < self.t3:
                # Shift growing
                if self.transition_type == "sigmoid":
                    S_shift = transition_sigmoid(t, (self.t2 + self.t3)/2, self.tau_shift)
                elif self.transition_type == "exponential":
                    S_shift = transition_exponential(t, self.t2, self.tau_shift)
                elif self.transition_type == "polynomial":
                    S_shift = transition_polynomial(t, self.t2, self.t3 - self.t2, n=3)
                else:
                    S_shift = (t - self.t2) / (self.t3 - self.t2)
            else:
                # Full shift achieved
                S_shift = 1.0
        else:  # overlapping mode
            # Shift grows with shell
            if t < self.t3:
                if self.transition_type == "sigmoid":
                    S_shift = transition_sigmoid(t, self.t3/2, self.t3/4)
                else:
                    S_shift = np.clip(t / self.t3, 0, 1)
            else:
                S_shift = 1.0

        # Spatial shape
        f_spatial = self.shape_function(r)

        # Direction: assume motion in +x direction
        # Magnitude: v_final * c * f(r) * S(t)
        v_magnitude = self.v_final * c * f_spatial * S_shift

        # Components
        beta_x = v_magnitude
        beta_y = np.zeros_like(x)
        beta_z = np.zeros_like(x)

        return beta_x, beta_y, beta_z

    def lapse_function(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        Lapse function alpha(r,t)

        For now, use simple approximation based on shell mass:
        alpha ≈ sqrt(1 - 2GM/(rc^2))

        Args:
            r: Radial distance
            t: Time

        Returns:
            Lapse function alpha
        """
        # Schwarzschild-like lapse from shell
        # Get current shell strength
        if t < self.t1:
            if self.transition_type == "sigmoid":
                S_shell = transition_sigmoid(t, self.t1/2, self.tau_shell)
            else:
                S_shell = np.clip(t / self.t1, 0, 1)
        else:
            S_shell = 1.0

        # Gravitational radius
        r_g = 2.0 * G * self.M / c**2

        # Lapse: alpha = sqrt(1 - r_g / r) inside shell region
        # Smooth transition
        f_shell = self.shape_function(r)

        # Simple model: lapse reduces inside shell proportional to mass
        alpha = np.ones_like(r)
        mask = r > r_g  # Avoid singularity
        alpha[mask] = np.sqrt(1.0 - S_shell * f_shell[mask] * r_g / r[mask])
        alpha[~mask] = 0.1  # Small value inside Schwarzschild radius

        # Ensure alpha >= 0.1 to avoid numerical issues
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
        R = np.maximum(R, 1e-10)  # Avoid r=0

        # Lapse function
        alpha = self.lapse_function(R, t)

        # Shift vector
        beta_x, beta_y, beta_z = self.shift_vector(X, Y, Z, R, t)

        # Shift vector dictionary (covariant)
        beta = {
            0: beta_x,
            1: beta_y,
            2: beta_z
        }

        # Spatial metric (flat for simplicity, could add shell-induced curvature)
        # For now: gamma_ij = delta_ij
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

        # Build metric from 3+1 components
        metric_dict = three_plus_one_builder(alpha, beta, gamma)

        # Create Tensor object
        metric = Tensor(
            metric_dict,
            tensor_type="metric",
            name=f"HybridMetric_t{t:.1f}",
            index="covariant",
            coords="cartesian",
            params={
                'R1': self.R1,
                'R2': self.R2,
                'M': self.M,
                'v_final': self.v_final,
                't': t,
                'stage_mode': self.stage_mode
            }
        )

        return metric


def run_hybrid_metrics_simulation(
    params: Optional[Dict] = None,
    grid_size: Tuple[int, int, int, int] = (20, 40, 40, 40),
    spatial_extent: float = 100.0,  # meters
    verbose: bool = True
) -> Dict:
    """
    Run full simulation of hybrid metrics approach

    Args:
        params: Dictionary of parameters (uses defaults if None)
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
            'M': 4.49e27,  # 2.365 Jupiter masses
            'v_final': 0.02,
            'sigma': 0.02,
            't1': 30.0,
            't2': 50.0,
            't3': 150.0,
            'transition_type': 'sigmoid',
            'stage_mode': 'sequential'
        }

    if verbose:
        print("=" * 70)
        print("APPROACH 3: HYBRID METRICS - Staged Acceleration")
        print("=" * 70)
        print(f"\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")
        print()

    # Create warp drive
    wd = HybridMetricWarpDrive(**params)

    # Create time-dependent metric
    time_range = (0.0, params['t3'] + 20.0)
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
        name="HybridMetrics"
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
        'approach': 'Hybrid Metrics (Approach 3)',
        'params': params,
        'results': results,
        'metrics': metrics,
        'warp_drive': wd,
        'time_dependent_metric': td_metric
    }

    return output


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING APPROACH 3: HYBRID METRICS")
    print("="*70 + "\n")

    # Test with default parameters
    results = run_hybrid_metrics_simulation(
        grid_size=(10, 20, 20, 20),  # Smaller grid for testing
        spatial_extent=50.0,
        verbose=True
    )

    print("\n" + "="*70)
    print("Test complete! Results stored in 'results' dictionary.")
    print("="*70)
