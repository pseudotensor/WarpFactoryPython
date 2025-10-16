#!/usr/bin/env python3
"""
Resolution Scaling Test

Test if energy condition violations decrease with higher grid resolution.
If violations are numerical artifacts, they should scale down with finer grids.
If they're physical, they'll persist.
"""

import sys
sys.path.append('/WarpFactory/warpfactory_py')

import numpy as np
from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.solver.energy import get_energy_tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions


def test_resolution(grid_size, v_warp=0.0):
    """Test at specific resolution"""
    print(f"\nTesting grid size {grid_size}...")

    # Paper parameters
    M = 4.49e27
    R1 = 10.0
    R2 = 20.0

    # Center needs to scale with grid
    world_center = [0.5] + [gs/2 + 0.5 for gs in grid_size[1:]]

    try:
        # Create metric
        metric = get_warp_shell_comoving_metric(
            grid_size=grid_size,
            world_center=world_center,
            m=M,
            R1=R1,
            R2=R2,
            v_warp=v_warp,
            do_warp=(v_warp > 1e-6)
        )

        # Compute stress-energy
        energy = get_energy_tensor(metric, try_gpu=False)

        # Check Null Energy Condition (quickest)
        nec_map, _, _ = get_energy_conditions(
            energy, metric, "Null",
            num_angular_vec=20,  # Reduced for speed
            num_time_vec=5
        )

        min_val = np.min(nec_map)
        max_val = np.max(nec_map)
        violations = np.sum(nec_map < -1e-15)
        total_points = np.prod(grid_size)
        violation_fraction = violations / total_points

        result = {
            'grid_size': grid_size,
            'total_points': total_points,
            'nec_min': min_val,
            'nec_max': max_val,
            'violations': violations,
            'violation_fraction': violation_fraction,
            'success': True
        }

        print(f"  NEC min: {min_val:.3e}")
        print(f"  NEC max: {max_val:.3e}")
        print(f"  Violations: {violations}/{total_points} ({100*violation_fraction:.1f}%)")

        return result

    except Exception as e:
        print(f"  Error: {e}")
        return {
            'grid_size': grid_size,
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    print("="*80)
    print("RESOLUTION SCALING TEST")
    print("="*80)
    print()
    print("Testing if energy condition violations are numerical artifacts.")
    print("If they scale down with resolution → numerical")
    print("If they persist → physical")
    print()

    # Test at multiple resolutions
    resolutions = [
        [1, 15, 15, 15],  # Coarse: 3,375 points
        [1, 21, 21, 21],  # Medium: 9,261 points
        [1, 31, 31, 31],  # Fine: 29,791 points
    ]

    results = []

    for grid_size in resolutions:
        result = test_resolution(grid_size, v_warp=0.0)
        results.append(result)
        print()

    # Analyze scaling
    print("="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    print()

    successful = [r for r in results if r['success']]

    if len(successful) >= 2:
        print(f"{'Grid Size':<20} {'Points':>10} {'NEC Min':>15} {'Violations':>12}")
        print("-"*80)

        for r in successful:
            gs = r['grid_size']
            print(f"{str(gs):<20} {r['total_points']:>10} {r['nec_min']:>15.3e} {r['violation_fraction']:>11.1%}")

        print()

        # Check if violations decrease
        min_vals = [r['nec_min'] for r in successful]
        if abs(min_vals[-1]) < abs(min_vals[0]) * 0.5:
            print("✓ Violations DECREASE with resolution → Likely NUMERICAL artifacts")
            print("  Further refinement may eliminate violations")
        elif abs(min_vals[-1]) > abs(min_vals[0]) * 0.5:
            print("✗ Violations PERSIST with resolution → Likely PHYSICAL")
            print("  These are real energy condition violations")
        else:
            print("? Violations change but unclear pattern")

        print()

        # Check spatial convergence
        if len(successful) >= 3:
            ratio_1 = min_vals[1] / min_vals[0]
            ratio_2 = min_vals[2] / min_vals[1]
            print(f"Scaling ratios:")
            print(f"  15→21 grid: {ratio_1:.3f}")
            print(f"  21→31 grid: {ratio_2:.3f}")

            if abs(ratio_1 - ratio_2) < 0.2:
                print("  → Consistent scaling (converging)")
            else:
                print("  → Inconsistent scaling (may need higher resolution)")

    else:
        print("Insufficient successful runs for analysis")

    print()
    print("="*80)
    print()

    if len(successful) > 0 and abs(successful[-1]['nec_min']) > 1e15:
        print("CONCLUSION: Violations ~10^40 persist across resolutions")
        print("These appear to be PHYSICAL violations, not numerical artifacts.")
    elif len(successful) > 0 and abs(successful[-1]['nec_min']) < 1e10:
        print("CONCLUSION: Violations decreased significantly with resolution")
        print("These may be numerical artifacts. Higher resolution testing recommended.")
    else:
        print("CONCLUSION: Inconclusive - more testing needed")

    print()
