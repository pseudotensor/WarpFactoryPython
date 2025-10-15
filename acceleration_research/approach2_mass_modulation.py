"""
Approach 2: Shell Mass Modulation

Varies shell mass M(t) and/or density profile ρ(r,t) during acceleration.

Physical Motivation: Energy density ρ provides "positive energy budget".
If ∂_t β creates negative contributions, can ∂_t ρ compensate?

Challenges: Where does extra mass come from? Could be:
- Gravitational binding energy conversion
- Kinetic to rest mass energy conversion
- External energy input (non-isolated system)

Expected Outcome: Unclear, possibly negative without physical mechanism.
This is exploratory/lower priority.
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


class MassModulationWarpDrive:
    """Warp drive with time-varying shell mass"""

    def __init__(
        self,
        R1: float = 10.0,
        R2: float = 20.0,
        M0: float = 4.49e27,  # Initial mass
        v_final: float = 0.02,
        sigma: float = 0.02,
        t0: float = 50.0,
        tau: float = 25.0,
        modulation_mode: str = "velocity_proportional",  # "velocity_proportional", "acceleration_proportional", "exponential"
        mass_amplitude: float = 0.3  # Fractional change in mass
    ):
        self.R1 = R1
        self.R2 = R2
        self.M0 = M0
        self.v_final = v_final
        self.sigma = sigma
        self.t0 = t0
        self.tau = tau
        self.modulation_mode = modulation_mode
        self.mass_amplitude = mass_amplitude

        self.R_center = (R1 + R2) / 2.0
        self.Delta_R = (R2 - R1) / 2.0

    def shape_function(self, r: np.ndarray) -> np.ndarray:
        inner_edge = np.tanh(self.sigma * self.Delta_R * (r - self.R1))
        outer_edge = np.tanh(self.sigma * self.Delta_R * (self.R2 - r))
        return 0.25 * (1.0 + inner_edge) * (1.0 + outer_edge)

    def temporal_transition(self, t: float) -> float:
        return transition_sigmoid(t, self.t0, self.tau)

    def mass_modulation(self, t: float) -> float:
        """Time-dependent mass M(t) = M0 * (1 + δM(t))"""
        S_t = self.temporal_transition(t)

        if self.modulation_mode == "velocity_proportional":
            # δM ∝ β²(t) - mass increases with kinetic energy
            delta_M = self.mass_amplitude * S_t**2

        elif self.modulation_mode == "acceleration_proportional":
            # δM ∝ dβ/dt - mass changes during active acceleration
            # Approximate derivative of sigmoid
            dS_dt = (1.0 / self.tau) * (1.0 - S_t**2)  # derivative of tanh
            delta_M = self.mass_amplitude * np.abs(dS_dt) * self.tau

        elif self.modulation_mode == "exponential":
            # M(t) = M0 * exp(α * ∫ β² dt)
            integrated = S_t * self.tau  # Approximate integral
            delta_M = self.mass_amplitude * (np.exp(S_t) - 1.0)

        else:
            delta_M = 0.0

        M_t = self.M0 * (1.0 + delta_M)
        return M_t

    def shell_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """Time-varying density"""
        f_spatial = self.shape_function(r)
        M_t = self.mass_modulation(t)

        volume_shell = 4.0 * np.pi * self.R_center**2 * self.Delta_R
        rho = (M_t / volume_shell) * f_spatial

        return rho

    def shift_vector(self, x, y, z, r, t):
        f_spatial = self.shape_function(r)
        S_t = self.temporal_transition(t)
        v_magnitude = self.v_final * c * f_spatial * S_t
        return v_magnitude, np.zeros_like(x), np.zeros_like(x)

    def lapse_function(self, r: np.ndarray, t: float) -> np.ndarray:
        """Lapse with time-dependent mass"""
        M_t = self.mass_modulation(t)
        r_g = 2.0 * G * M_t / c**2
        f_shell = self.shape_function(r)

        alpha = np.ones_like(r)
        mask = r > r_g
        alpha[mask] = np.sqrt(1.0 - f_shell[mask] * r_g / r[mask])
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
            name=f"MassModulation_t{t:.1f}",
            index="covariant",
            coords="cartesian",
            params={'M_t': self.mass_modulation(t), 't': t}
        )


def run_mass_modulation_simulation(params=None, grid_size=(20, 40, 40, 40),
                                   spatial_extent=100.0, verbose=True):
    if params is None:
        params = {
            'R1': 10.0, 'R2': 20.0, 'M0': 4.49e27, 'v_final': 0.02,
            'sigma': 0.02, 't0': 50.0, 'tau': 25.0,
            'modulation_mode': 'velocity_proportional',
            'mass_amplitude': 0.3
        }

    if verbose:
        print("=" * 70)
        print("APPROACH 2: SHELL MASS MODULATION")
        print("=" * 70)
        print(f"\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

    wd = MassModulationWarpDrive(**params)

    time_range = (0.0, params['t0'] + 4*params['tau'])
    spatial_extent_tuple = [(-spatial_extent, spatial_extent)] * 3

    td_metric = TimeDependentMetric(
        grid_size=list(grid_size),
        time_range=time_range,
        spatial_extent=spatial_extent_tuple,
        metric_function=lambda t, X, Y, Z: wd.get_metric_tensor(t, X, Y, Z),
        name="MassModulation"
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
        'approach': 'Mass Modulation (Approach 2)',
        'params': params,
        'results': results,
        'metrics': metrics,
        'warp_drive': wd,
        'time_dependent_metric': td_metric
    }


if __name__ == "__main__":
    print("\nTESTING APPROACH 2: MASS MODULATION\n")
    results = run_mass_modulation_simulation(
        grid_size=(10, 20, 20, 20),
        spatial_extent=50.0,
        verbose=True
    )
    print("\nTest complete!")
