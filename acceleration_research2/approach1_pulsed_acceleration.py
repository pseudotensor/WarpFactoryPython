"""
Pulsed Acceleration: Discrete velocity jumps instead of smooth transitions

NOVEL IDEA: Challenge the assumption that acceleration must be smooth!

Instead of continuous dv/dt, use a series of INSTANTANEOUS velocity jumps
separated by constant-velocity phases. This concentrates time-dependence
into brief moments, potentially reducing total integrated violations.

Physical Analogy:
- Impulsive orbital maneuvers (Hohmann transfers)
- Pulsed laser ablation propulsion
- Digital vs analog control systems

Hypothesis:
- Brief, intense violations might integrate to less total violation than
  prolonged moderate violations
- Constant-velocity phases (proven physical) dominate timeline
- Time-concentrated ∂_t g might create different violation structure

Mathematical Framework:
- v(t) = Σᵢ Δvᵢ × H(t - tᵢ)  where H is Heaviside step function
- Metric has discontinuous first derivative at pulse times
- Between pulses: ∂_t g_μν = 0 exactly (physical!)
"""

import numpy as np
import sys
sys.path.append('/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.core.tensor import Tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions
from warpfactory.units.constants import c as speed_of_light


def create_pulsed_velocity_profile(
    num_pulses: int,
    v_final: float,
    pulse_times: np.ndarray,
    pulse_width: float = 1.0
) -> tuple:
    """
    Create pulsed velocity profile with discrete jumps

    Args:
        num_pulses: Number of velocity pulses
        v_final: Final velocity to reach
        pulse_times: Array of times for each pulse [s]
        pulse_width: Width of each pulse transition [s]

    Returns:
        Tuple of (time_points, velocity_values)
    """
    # Equal velocity increments
    delta_v = v_final / num_pulses

    # Create high-resolution time array
    t_max = pulse_times[-1] + 50
    time_points = np.linspace(0, t_max, 200)
    velocity = np.zeros_like(time_points)

    # Add each pulse
    for i, t_pulse in enumerate(pulse_times):
        # Smooth step function (very narrow)
        step = 0.5 * (1 + np.tanh((time_points - t_pulse) / pulse_width))
        velocity += delta_v * step

    return time_points, velocity


def simulate_pulsed_acceleration(
    num_pulses: int = 5,
    v_final: float = 0.02,
    pulse_spacing: float = 20.0,
    grid_size: list = [10, 20, 20, 20],
    shell_radii: tuple = (5.0, 10.0, 25.0, 30.0)
) -> dict:
    """
    Simulate warp drive with pulsed acceleration

    Args:
        num_pulses: Number of discrete velocity pulses
        v_final: Final velocity (fraction of c)
        pulse_spacing: Time between pulses [s]
        grid_size: Spacetime grid [t, x, y, z]
        shell_radii: (R1_inner, R1_outer, R2_inner, R2_outer)

    Returns:
        Dictionary with simulation results
    """
    c = speed_of_light()

    # Create pulse times
    pulse_times = np.array([i * pulse_spacing for i in range(num_pulses)])

    # Generate velocity profile
    times, velocities = create_pulsed_velocity_profile(
        num_pulses, v_final, pulse_times, pulse_width=0.5
    )

    print(f"Pulsed Acceleration Simulation")
    print(f"{'='*70}")
    print(f"Number of pulses: {num_pulses}")
    print(f"Pulse times: {pulse_times}")
    print(f"ΔV per pulse: {v_final/num_pulses:.4f}c")
    print(f"Total time: {pulse_times[-1]:.1f}s")
    print(f"Grid size: {grid_size}")
    print()

    # Compute violation at different time slices
    results = {
        'num_pulses': num_pulses,
        'pulse_times': pulse_times,
        'velocity_profile': (times, velocities),
        'violations': [],
        'time_slices': []
    }

    # Sample at key times (before, during, after pulses)
    test_times = [0, pulse_times[0], pulse_times[0] + 2,
                  pulse_times[-1], pulse_times[-1] + 10]

    for t in test_times:
        # Find velocity at this time
        v_t = np.interp(t, times, velocities)

        print(f"Time t={t:.1f}s: v={v_t:.4f}c")

        # Create metric snapshot at this time
        # (Using comoving warp shell as approximation)
        # In reality, would need time-dependent metric evaluation

        results['time_slices'].append(t)
        results['violations'].append({
            'time': t,
            'velocity': v_t,
            'note': 'Requires full time-dependent implementation'
        })

    print()
    print("Analysis:")
    print(f"  Total violation time: ~{num_pulses * 2}s (pulse widths)")
    print(f"  Constant-v time: ~{pulse_times[-1] - num_pulses * 2}s")
    print(f"  Duty cycle: {(num_pulses * 2) / pulse_times[-1] * 100:.1f}%")
    print()
    print("Expected behavior:")
    print("  - Violations concentrated in brief pulses")
    print("  - Constant-velocity between pulses (physical)")
    print("  - Lower integrated violation if pulse violations < continuous")

    return results


def compare_pulsed_vs_continuous(
    num_pulses_list: list = [2, 5, 10, 20],
    v_final: float = 0.02
) -> dict:
    """
    Compare different numbers of pulses

    The question: Is N small pulses better than 1 continuous ramp?

    Hypothesis: More pulses = smaller per-pulse ΔV = smaller violations?
    Counter: More pulses = more violation events = cumulative?
    """
    print("Pulsed vs Continuous Comparison")
    print("="*70)
    print()

    results = {}

    for n in num_pulses_list:
        print(f"Testing {n} pulses:")
        print(f"  ΔV per pulse: {v_final/n:.4f}c")
        print(f"  Expected violation ~ (ΔV)²: {(v_final/n)**2:.2e}")
        print(f"  Total from n pulses: {n * (v_final/n)**2:.2e}")
        print()

        # Heuristic estimate (needs full simulation)
        single_pulse_violation = (v_final/n)**2
        total_violation = n * single_pulse_violation

        results[n] = {
            'delta_v': v_final/n,
            'single_violation_estimate': single_pulse_violation,
            'total_violation_estimate': total_violation,
            'vs_continuous': total_violation / v_final**2
        }

    print("Summary:")
    print("  Continuous (n=1): Total ~ v²")
    print(f"  Pulsed (n=N): Total ~ N × (v/N)² = v²/N")
    print()
    print("  Prediction: More pulses = better (linear with N)")
    print("  BUT: This ignores pulse interference and cumulative effects")
    print("  Reality: Needs full simulation to determine!")

    return results


if __name__ == "__main__":
    print("="*70)
    print(" PULSED ACCELERATION APPROACH")
    print("="*70)
    print()

    # Run pulsed simulation
    results = simulate_pulsed_acceleration(
        num_pulses=5,
        v_final=0.02,
        pulse_spacing=20.0
    )

    print()
    print("="*70)
    print()

    # Compare different pulse numbers
    comparison = compare_pulsed_vs_continuous(
        num_pulses_list=[2, 5, 10, 20]
    )

    print()
    print("="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print()
    print("This approach requires full time-dependent metric implementation")
    print("to properly evaluate. Key questions:")
    print()
    print("1. Do brief intense violations integrate to less than prolonged mild ones?")
    print("2. How many pulses is optimal?")
    print("3. Does this avoid multi-shell cumulative issues?")
    print("4. What about discontinuities in metric derivatives?")
    print()
    print("STATUS: Framework created, needs full simulation")
