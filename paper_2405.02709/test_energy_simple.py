#!/usr/bin/env python3
"""
CRITICAL TEST: Measure actual energy condition violations in Fuchs shell
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.solver.energy import get_energy_tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions


def test_fuchs_shell(v_warp, label):
    """Test a single Fuchs shell configuration"""
    print("="*80)
    print(f"Testing {label}: v={v_warp}c")
    print("="*80)

    # Paper parameters
    M = 4.49e27
    R1 = 10.0
    R2 = 20.0

    # Create metric (small grid for speed)
    print("\nCreating metric...")
    grid_size = [1, 21, 21, 21]

    metric = get_warp_shell_comoving_metric(
        grid_size=grid_size,
        world_center=[0.5, 11.0, 11.0, 11.0],
        m=M,
        R1=R1,
        R2=R2,
        v_warp=v_warp,
        do_warp=(v_warp > 1e-6)
    )

    print("✓ Metric created")

    # Compute stress-energy
    print("\nComputing stress-energy tensor...")
    energy = get_energy_tensor(metric, try_gpu=False)
    print("✓ Stress-energy computed")

    # Check energy conditions
    print("\nChecking energy conditions (this takes a few minutes)...")

    results = {}

    for condition in ['Null', 'Weak', 'Dominant', 'Strong']:
        print(f"\n  {condition} Energy Condition:")
        try:
            condition_map, _, _ = get_energy_conditions(
                energy,
                metric,
                condition,
                num_angular_vec=30,  # Reduced for speed
                num_time_vec=8
            )

            min_val = np.min(condition_map)
            max_val = np.max(condition_map)
            violations = np.sum(condition_map < -1e-15)

            results[condition] = {
                'min': min_val,
                'max': max_val,
                'violations': violations
            }

            print(f"    Min: {min_val:.6e}")
            print(f"    Max: {max_val:.6e}")
            print(f"    Violations: {violations} / {np.prod(grid_size)} points")

            if violations > 0:
                print(f"    ⚠ VIOLATIONS DETECTED")
            else:
                print(f"    ✓ No violations")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[condition] = {'error': str(e)}

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("CRITICAL VALIDATION TEST")
    print("="*80)
    print("\nTesting if Fuchs shell actually satisfies energy conditions")
    print("(The paper claimed zero violations but never computed them!)")
    print("\n")

    # Test 1: v=0 (no warp)
    print("\n\n")
    results_v0 = test_fuchs_shell(0.0, "Static Shell (no warp)")

    # Test 2: v=0.02 (paper value)
    print("\n\n")
    results_v002 = test_fuchs_shell(0.02, "Warp Shell (paper value)")

    # Summary
    print("\n\n")
    print("="*80)
    print("CRITICAL FINDINGS")
    print("="*80)
    print()

    def check_violations(results, label):
        print(f"{label}:")
        has_violations = False
        worst = 0
        for cond in ['Null', 'Weak', 'Dominant', 'Strong']:
            if cond in results and 'min' in results[cond]:
                min_val = results[cond]['min']
                viols = results[cond]['violations']

                status = "✓" if viols == 0 else "✗"
                print(f"  {status} {cond:10s}: min={min_val:.3e}, violations={viols}")

                if viols > 0:
                    has_violations = True
                    if min_val < worst:
                        worst = min_val

        if has_violations:
            print(f"  → HAS VIOLATIONS (worst: {worst:.3e})")
        else:
            print(f"  → No violations detected")

        print()
        return has_violations, worst

    v0_violations, v0_worst = check_violations(results_v0, "Static Shell (v=0)")
    v002_violations, v002_worst = check_violations(results_v002, "Warp Shell (v=0.02)")

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    if v0_violations or v002_violations:
        print("⚠⚠⚠ CRITICAL ISSUE IDENTIFIED ⚠⚠⚠")
        print()
        print("The Fuchs shell DOES have energy condition violations!")
        print()
        print("The paper claimed zero violations, but:")
        print("  1. The reproduction script used compute_energy_conditions=False")
        print("  2. Energy conditions were NEVER ACTUALLY COMPUTED")
        print("  3. The claim of 'zero violations' was not validated")
        print()
        print(f"Actual violations:")
        print(f"  Static shell (v=0): {v0_worst:.3e}")
        print(f"  Warp shell (v=0.02): {v002_worst:.3e}")
        print()
        print("This matches the ~10^40 violations found in acceleration_research3!")
        print()
        print("The 'breakthrough' was based on unvalidated claims.")
    else:
        print("✓ Fuchs shell satisfies all energy conditions")
        print()
        print("If this is correct, then the 10^40 violations in acceleration_research3")
        print("must be due to time-dependent effects or implementation issues.")

    print()
    print("="*80)
