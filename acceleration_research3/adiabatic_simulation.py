#!/usr/bin/env python3
"""
Adiabatic Fuchs Shell Evolution - Full Simulation

Test if slowly evolving the Fuchs shell shift parameter can achieve
zero (or near-zero) energy condition violations during acceleration.

Key hypothesis: violations ~ (dv/dt)² → 0 as T → ∞
"""

import numpy as np
import sys
import time
sys.path.append('/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.solver.energy import get_energy_tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions


def smooth_transition(t, T, transition_type='tanh'):
    """
    Smooth transition function from 0 to 1

    Args:
        t: Time
        T: Total transition duration
        transition_type: 'tanh', 'sigmoid', or 'polynomial'

    Returns:
        Value between 0 and 1
    """
    tau = t / T  # Normalized time

    if transition_type == 'tanh':
        # Hyperbolic tangent (smooth, symmetric)
        return 0.5 * (1 + np.tanh(6 * (tau - 0.5)))
    elif transition_type == 'sigmoid':
        # Logistic function
        return 1 / (1 + np.exp(-12 * (tau - 0.5)))
    elif transition_type == 'polynomial':
        # 5th order polynomial (smooth first and second derivatives)
        if tau <= 0:
            return 0.0
        elif tau >= 1:
            return 1.0
        else:
            return 6*tau**5 - 15*tau**4 + 10*tau**3
    else:
        return tau  # Linear (for comparison)


def simulate_adiabatic_acceleration(
    v_final=0.02,
    T_accel=100.0,
    num_time_steps=20,
    grid_size=[1, 21, 21, 21],
    compute_full_conditions=False
):
    """
    Simulate adiabatic acceleration of Fuchs warp shell

    Args:
        v_final: Final velocity (fraction of c)
        T_accel: Total acceleration time [s]
        num_time_steps: Number of time slices to sample
        grid_size: Spatial grid resolution
        compute_full_conditions: Whether to compute energy conditions (slow)

    Returns:
        Dictionary with results
    """
    print("="*80)
    print(" ADIABATIC FUCHS SHELL ACCELERATION SIMULATION")
    print("="*80)
    print()

    print(f"Parameters:")
    print(f"  Final velocity: {v_final}c")
    print(f"  Acceleration time: {T_accel} s")
    print(f"  Time steps: {num_time_steps}")
    print(f"  Grid size: {grid_size}")
    print(f"  dv/dt: {v_final/T_accel:.2e} c/s")
    print()

    # Shell parameters (from Fuchs paper)
    M = 4.49e27  # kg (2.365 Jupiter masses)
    R1 = 10.0    # m
    R2 = 20.0    # m

    # Characteristic timescale
    c_val = 2.99792458e8
    tau_char = R2 / c_val
    adiabatic_param = T_accel / tau_char

    print(f"Characteristic timescale: {tau_char:.2e} s")
    print(f"Adiabatic parameter: T/τ = {adiabatic_param:.2e}")

    if adiabatic_param > 1000:
        print(f"  → STRONGLY ADIABATIC ✓")
    elif adiabatic_param > 10:
        print(f"  → ADIABATIC")
    else:
        print(f"  → NON-ADIABATIC (too fast)")
    print()

    # Time evolution
    times = np.linspace(0, T_accel, num_time_steps)
    results = {
        'times': times,
        'velocities': [],
        'violations': {'null': [], 'weak': [], 'dominant': [], 'strong': []},
        'energy_density_max': [],
        'energy_density_min': []
    }

    print("Time evolution:")
    print()

    for i, t in enumerate(times):
        # Velocity at this time
        S_t = smooth_transition(t, T_accel, 'tanh')
        v_t = v_final * S_t
        results['velocities'].append(v_t)

        print(f"  Step {i+1}/{num_time_steps}: t={t:.1f}s, v={v_t:.4f}c", end="")

        try:
            # Create Fuchs shell at this velocity
            metric = get_warp_shell_comoving_metric(
                grid_size=grid_size,
                world_center=[0.5] + [gs/2 + 0.5 for gs in grid_size[1:]],
                m=M,
                R1=R1,
                R2=R2,
                v_warp=v_t,
                do_warp=(v_t > 1e-6)
            )

            if compute_full_conditions and i % 5 == 0:  # Sample every 5th step
                print(" [computing energy conditions...]", end="", flush=True)

                # Compute energy tensor
                energy = get_energy_tensor(metric, try_gpu=False)

                # Energy density
                rho = energy[(0, 0)]
                results['energy_density_max'].append(np.max(rho))
                results['energy_density_min'].append(np.min(rho))

                # Energy conditions (expensive!)
                null_map, _, _ = get_energy_conditions(
                    energy, metric, "Null",
                    num_angular_vec=20, num_time_vec=5
                )

                results['violations']['null'].append(np.min(null_map))
                print(f" NEC_min={np.min(null_map):.2e}", end="")

            print(" ✓")

        except Exception as e:
            print(f" ✗ Error: {e}")
            results['violations']['null'].append(None)

    print()
    return results


def analyze_scaling_law(results):
    """Analyze if violations follow (dv/dt)² scaling"""
    print("="*80)
    print(" SCALING LAW ANALYSIS")
    print("="*80)
    print()

    times = results['times']
    velocities = results['velocities']

    if len(results['violations']['null']) > 0 and results['violations']['null'][0] is not None:
        violations = [v for v in results['violations']['null'] if v is not None]

        print("Violation vs time:")
        for i, (t, viol) in enumerate(zip(times[::5], violations)):
            print(f"  t={t:.1f}s: Violation = {viol:.2e}")

        print()
        print("Scaling check:")
        # Check if violations decrease with slower acceleration
        print("  (Needs multiple simulations with different T to verify)")
    else:
        print("Energy conditions not computed (set compute_full_conditions=True)")

    print()


if __name__ == "__main__":
    print("Testing adiabatic acceleration at different timescales...")
    print()

    # Test quick (without energy conditions)
    print("QUICK TEST (no energy conditions):")
    results_quick = simulate_adiabatic_acceleration(
        v_final=0.02,
        T_accel=100.0,
        num_time_steps=10,
        grid_size=[1, 11, 11, 11],
        compute_full_conditions=False
    )

    print()
    print("="*80)
    print()
    print("For FULL simulation with energy condition verification:")
    print("Run with compute_full_conditions=True (takes ~5-10 minutes)")
    print()
    print("Expected results if adiabatic hypothesis is correct:")
    print(f"  T=100s: violations ~ {(0.02/100)**2:.2e}")
    print(f"  T=1000s: violations ~ {(0.02/1000)**2:.2e}")
    print(f"  T=10000s: violations ~ {(0.02/10000)**2:.2e} (EFFECTIVE ZERO!)")
    print()
    print("STATUS: Framework ready for full testing")
