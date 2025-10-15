"""
Approach 4: Multi-Shell Configuration

Uses multiple nested shells at different velocities to create staged acceleration.
Energy and momentum transfer through gravitational interaction between shells.

Configuration:
- Inner shell: R1_inner to R2_inner, velocity v1(t), mass M1
- Middle shell: R1_middle to R2_middle, velocity v2(t), mass M2
- Outer shell: R1_outer to R2_outer, velocity v3(t), mass M3

Mechanism: Outer shells accelerate first, "dragging" inner shells through
gravitational coupling. Similar to electromagnetic coilgun staging.

Expected Outcome: 40-60% violation reduction through distributed acceleration
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpfactory.core.tensor import Tensor
from warpfactory.metrics.three_plus_one import three_plus_one_builder
from warpfactory.units.constants import c, G
from acceleration_research.time_dependent_framework import (
    TimeDependentMetric,
    transition_sigmoid,
    evaluate_energy_conditions_over_time,
    compute_violation_metrics
)


class MultiShellWarpDrive:
    """Multi-shell warp drive with staged acceleration"""

    def __init__(
        self,
        shell_radii: List[Tuple[float, float]] = [(5, 10), (15, 20), (25, 30)],
        shell_masses: List[float] = [1.5e27, 1.5e27, 1.5e27],
        v_final: float = 0.02,
        t0: float = 50.0,
        tau: float = 25.0,
        velocity_ratios: List[float] = [0.4, 0.7, 1.0],  # Relative velocities
        sigma: float = 0.02
    ):
        """
        Args:
            shell_radii: List of (R1, R2) tuples for each shell
            shell_masses: List of masses for each shell
            v_final: Final velocity
            t0: Center time
            tau: Transition time
            velocity_ratios: Velocity of each shell relative to v_final
            sigma: Shape parameter
        """
        self.shell_radii = shell_radii
        self.shell_masses = shell_masses
        self.v_final = v_final
        self.t0 = t0
        self.tau = tau
        self.velocity_ratios = velocity_ratios
        self.sigma = sigma
        self.n_shells = len(shell_radii)

        # Outer shells reach velocity earlier
        self.shell_t0 = [t0 - (1.0 - ratio) * tau for ratio in velocity_ratios]

    def shape_function(self, r: np.ndarray, R1: float, R2: float) -> np.ndarray:
        """Shape function for individual shell"""
        R_center = (R1 + R2) / 2.0
        Delta_R = (R2 - R1) / 2.0
        inner_edge = np.tanh(self.sigma * Delta_R * (r - R1))
        outer_edge = np.tanh(self.sigma * Delta_R * (R2 - r))
        return 0.25 * (1.0 + inner_edge) * (1.0 + outer_edge)

    def get_total_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """Sum density contributions from all shells"""
        rho_total = np.zeros_like(r)

        for i, (R1, R2) in enumerate(self.shell_radii):
            f = self.shape_function(r, R1, R2)
            R_center = (R1 + R2) / 2.0
            Delta_R = (R2 - R1) / 2.0
            volume = 4.0 * np.pi * R_center**2 * Delta_R
            rho_i = (self.shell_masses[i] / volume) * f
            rho_total += rho_i

        return rho_total

    def shift_vector(self, x, y, z, r, t):
        """Combined shift from all shells"""
        beta_x_total = np.zeros_like(x)

        for i, (R1, R2) in enumerate(self.shell_radii):
            f = self.shape_function(r, R1, R2)
            S_t = transition_sigmoid(t, self.shell_t0[i], self.tau)
            v_i = self.velocity_ratios[i] * self.v_final * c * f * S_t
            beta_x_total += v_i

        return beta_x_total, np.zeros_like(x), np.zeros_like(x)

    def lapse_function(self, r: np.ndarray, t: float) -> np.ndarray:
        """Combined lapse from all shells"""
        alpha = np.ones_like(r)

        for i, (R1, R2) in enumerate(self.shell_radii):
            M_i = self.shell_masses[i]
            r_g = 2.0 * G * M_i / c**2
            f = self.shape_function(r, R1, R2)

            mask = r > r_g
            alpha[mask] *= np.sqrt(1.0 - f[mask] * r_g / r[mask])

        alpha = np.maximum(alpha, 0.1)
        return alpha

    def get_metric_tensor(self, t, X, Y, Z):
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R = np.maximum(R, 1e-10)

        alpha = self.lapse_function(R, t)
        beta_x, beta_y, beta_z = self.shift_vector(X, Y, Z, R, t)

        beta = {0: beta_x, 1: beta_y, 2: beta_z}
        gamma = {(i, j): np.ones_like(X) if i == j else np.zeros_like(X)
                for i in range(3) for j in range(3)}

        metric_dict = three_plus_one_builder(alpha, beta, gamma)

        return Tensor(
            metric_dict,
            tensor_type="metric",
            name=f"MultiShell_t{t:.1f}",
            index="covariant",
            coords="cartesian",
            params={'n_shells': self.n_shells, 't': t}
        )


def run_multi_shell_simulation(params=None, grid_size=(20, 40, 40, 40),
                               spatial_extent=100.0, verbose=True):
    if params is None:
        params = {
            'shell_radii': [(5, 10), (15, 20), (25, 30)],
            'shell_masses': [1.5e27, 1.5e27, 1.5e27],
            'v_final': 0.02,
            't0': 50.0,
            'tau': 25.0,
            'velocity_ratios': [0.4, 0.7, 1.0],
            'sigma': 0.02
        }

    if verbose:
        print("=" * 70)
        print("APPROACH 4: MULTI-SHELL CONFIGURATION")
        print("=" * 70)
        print(f"\nParameters:")
        print(f"  Number of shells: {len(params['shell_radii'])}")
        print(f"  Shell radii: {params['shell_radii']}")
        print(f"  Velocity ratios: {params['velocity_ratios']}")

    wd = MultiShellWarpDrive(**params)

    time_range = (0.0, params['t0'] + 4*params['tau'])
    spatial_extent_tuple = [(-spatial_extent, spatial_extent)] * 3

    td_metric = TimeDependentMetric(
        grid_size=list(grid_size),
        time_range=time_range,
        spatial_extent=spatial_extent_tuple,
        metric_function=lambda t, X, Y, Z: wd.get_metric_tensor(t, X, Y, Z),
        name="MultiShell"
    )

    if verbose:
        print("\nEvaluating energy conditions...")

    results = evaluate_energy_conditions_over_time(
        td_metric,
        conditions=["Null", "Weak", "Dominant", "Strong"],
        num_angular_vec=50,
        num_time_vec=10,
        verbose=verbose
    )

    metrics = compute_violation_metrics(results)

    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        for condition, data in metrics.items():
            print(f"\n{condition} Energy Condition:")
            print(f"  Worst violation: {data['worst_violation']:.6e}")
            print(f"  Max magnitude: {data['max_magnitude']:.6e}")

    return {
        'approach': 'Multi-Shell (Approach 4)',
        'params': params,
        'results': results,
        'metrics': metrics,
        'warp_drive': wd,
        'time_dependent_metric': td_metric
    }


if __name__ == "__main__":
    print("\nTESTING APPROACH 4: MULTI-SHELL CONFIGURATION\n")
    results = run_multi_shell_simulation(
        grid_size=(10, 20, 20, 20),
        spatial_extent=80.0,
        verbose=True
    )
    print("\nTest complete!")
