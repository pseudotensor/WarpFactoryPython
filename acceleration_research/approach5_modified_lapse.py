"""
Approach 5: Modified Lapse Functions - Optimization Approach

This approach uses time-dependent lapse function α(r,t) during acceleration
to help absorb energy from the time-varying shift vector.

Concept: α relates proper time to coordinate time (dτ = α dt). By making α
vary in time and space during acceleration, we create a "time gradient" that
might help satisfy energy conditions.

Physical Motivation: Paper 2405.02709 states "Changed lapse rate creates a
Shapiro time delay" and "This constraint may be another important aspect of
physicality."

Expected Outcome: 10-40% violation reduction through optimization
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
    evaluate_energy_conditions_over_time,
    compute_violation_metrics
)


class ModifiedLapseWarpDrive:
    """
    Warp drive with time-dependent lapse function during acceleration
    """

    def __init__(
        self,
        R1: float = 10.0,
        R2: float = 20.0,
        M: float = 4.49e27,
        v_final: float = 0.02,
        sigma: float = 0.02,
        t0: float = 50.0,
        tau: float = 25.0,
        lapse_mode: str = "velocity_coupled",  # "static", "velocity_coupled", "spatial_gradient"
        lapse_amplitude: float = 0.2  # Max change in lapse
    ):
        self.R1 = R1
        self.R2 = R2
        self.M = M
        self.v_final = v_final
        self.sigma = sigma
        self.t0 = t0
        self.tau = tau
        self.lapse_mode = lapse_mode
        self.lapse_amplitude = lapse_amplitude

        self.R_center = (R1 + R2) / 2.0
        self.Delta_R = (R2 - R1) / 2.0

    def shape_function(self, r: np.ndarray) -> np.ndarray:
        inner_edge = np.tanh(self.sigma * self.Delta_R * (r - self.R1))
        outer_edge = np.tanh(self.sigma * self.Delta_R * (self.R2 - r))
        return 0.25 * (1.0 + inner_edge) * (1.0 + outer_edge)

    def temporal_transition(self, t: float) -> float:
        return transition_sigmoid(t, self.t0, self.tau)

    def shift_vector(self, x, y, z, r, t):
        f_spatial = self.shape_function(r)
        S_t = self.temporal_transition(t)
        v_magnitude = self.v_final * c * f_spatial * S_t
        return v_magnitude, np.zeros_like(x), np.zeros_like(x)

    def lapse_function(self, x, y, z, r, t):
        """Time-dependent lapse function"""
        # Base lapse from shell
        r_g = 2.0 * G * self.M / c**2
        f_shell = self.shape_function(r)

        alpha_base = np.ones_like(r)
        mask = r > r_g
        alpha_base[mask] = np.sqrt(1.0 - f_shell[mask] * r_g / r[mask])
        alpha_base = np.maximum(alpha_base, 0.1)

        # Time-dependent modification
        if self.lapse_mode == "static":
            # No time dependence
            alpha = alpha_base

        elif self.lapse_mode == "velocity_coupled":
            # Lapse changes with velocity: α(t) = α₀(1 - δα·β²(t))
            S_t = self.temporal_transition(t)
            beta_squared = (self.v_final * S_t)**2
            delta_alpha = self.lapse_amplitude * beta_squared * f_shell
            alpha = alpha_base * (1.0 - delta_alpha)

        elif self.lapse_mode == "spatial_gradient":
            # Front-back asymmetry: time runs faster ahead, slower behind
            S_t = self.temporal_transition(t)
            # Gradient along x-direction (motion direction)
            x_normalized = x / (self.R_center)
            gradient_factor = self.lapse_amplitude * S_t * x_normalized * f_shell
            alpha = alpha_base * (1.0 + gradient_factor)

        else:
            alpha = alpha_base

        # Ensure positive and bounded
        alpha = np.clip(alpha, 0.1, 2.0)

        return alpha

    def get_metric_tensor(self, t, X, Y, Z):
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R = np.maximum(R, 1e-10)

        alpha = self.lapse_function(X, Y, Z, R, t)
        beta_x, beta_y, beta_z = self.shift_vector(X, Y, Z, R, t)

        beta = {0: beta_x, 1: beta_y, 2: beta_z}
        gamma = {(i, j): np.ones_like(X) if i == j else np.zeros_like(X)
                for i in range(3) for j in range(3)}

        metric_dict = three_plus_one_builder(alpha, beta, gamma)

        return Tensor(
            metric_dict,
            tensor_type="metric",
            name=f"ModifiedLapse_t{t:.1f}",
            index="covariant",
            coords="cartesian",
            params={'R1': self.R1, 'R2': self.R2, 'M': self.M, 't': t,
                   'lapse_mode': self.lapse_mode}
        )


def run_modified_lapse_simulation(params=None, grid_size=(20, 40, 40, 40),
                                  spatial_extent=100.0, verbose=True):
    if params is None:
        params = {
            'R1': 10.0, 'R2': 20.0, 'M': 4.49e27, 'v_final': 0.02,
            'sigma': 0.02, 't0': 50.0, 'tau': 25.0,
            'lapse_mode': 'velocity_coupled', 'lapse_amplitude': 0.2
        }

    if verbose:
        print("=" * 70)
        print("APPROACH 5: MODIFIED LAPSE FUNCTIONS")
        print("=" * 70)
        print(f"\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

    wd = ModifiedLapseWarpDrive(**params)

    time_range = (0.0, params['t0'] + 4*params['tau'])
    spatial_extent_tuple = [(-spatial_extent, spatial_extent)] * 3

    td_metric = TimeDependentMetric(
        grid_size=list(grid_size),
        time_range=time_range,
        spatial_extent=spatial_extent_tuple,
        metric_function=lambda t, X, Y, Z: wd.get_metric_tensor(t, X, Y, Z),
        name="ModifiedLapse"
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
            print(f"  L2 norm: {data['total_violation_L2']:.6e}")

    return {
        'approach': 'Modified Lapse (Approach 5)',
        'params': params,
        'results': results,
        'metrics': metrics,
        'warp_drive': wd,
        'time_dependent_metric': td_metric
    }


if __name__ == "__main__":
    print("\nTESTING APPROACH 5: MODIFIED LAPSE FUNCTIONS\n")
    results = run_modified_lapse_simulation(
        grid_size=(10, 20, 20, 20),
        spatial_extent=50.0,
        verbose=True
    )
    print("\nTest complete!")
