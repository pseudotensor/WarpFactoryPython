#!/usr/bin/env python3
"""
Critical Test: Run paper reproduction WITH energy conditions to verify claims.

This script will actually compute energy conditions that were skipped in the
original reproduction, revealing the true violation status.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduce_results import PaperReproduction
import numpy as np


def test_static_shells():
    """Test both static v=0 and v=0.02 shells with full energy conditions."""

    print("="*80)
    print("CRITICAL VALIDATION TEST")
    print("="*80)
    print()
    print("Testing Fuchs shell WITH actual energy condition computation")
    print("This was SKIPPED in the original reproduction (compute_energy_conditions=False)")
    print()

    reproduction = PaperReproduction()

    # Test 1: Static shell at v=0 (no warp)
    print("\n" + "="*80)
    print("TEST 1: Static Matter Shell (v=0, no warp)")
    print("="*80)

    shell_metric = reproduction.create_shell_metric()

    print("\nComputing stress-energy tensor...")
    shell_stress = reproduction.compute_stress_energy(shell_metric, "Static Shell v=0")

    print("\nComputing energy conditions (this is expensive, please wait)...")
    shell_conditions = reproduction.check_energy_conditions_full(
        shell_metric, shell_stress, "Static Shell v=0"
    )

    # Analyze results
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS - Static Shell (v=0):")
    print("-"*80)

    for condition_name in ['NEC', 'WEC', 'SEC', 'DEC']:
        if condition_name in shell_conditions:
            values = shell_conditions[condition_name]['values']
            min_val = np.min(values)
            max_val = np.max(values)
            violations = np.sum(values < -1e-15)  # Allow numerical precision

            print(f"\n{condition_name}:")
            print(f"  Min value: {min_val:.6e}")
            print(f"  Max value: {max_val:.6e}")
            print(f"  Violations (< -1e-15): {violations}")

            if violations > 0:
                print(f"  ⚠ VIOLATIONS DETECTED!")
                # Show where violations occur
                violation_indices = np.where(values < -1e-15)
                print(f"  Number of grid points with violations: {len(violation_indices[0])}")
                print(f"  Worst violation: {min_val:.6e}")
            else:
                print(f"  ✓ No violations")

    # Test 2: Static warp shell at v=0.02
    print("\n\n" + "="*80)
    print("TEST 2: Static Warp Shell (v=0.02)")
    print("="*80)

    warp_metric = reproduction.create_warp_shell_metric()

    print("\nComputing stress-energy tensor...")
    warp_stress = reproduction.compute_stress_energy(warp_metric, "Static Warp Shell v=0.02")

    print("\nComputing energy conditions...")
    warp_conditions = reproduction.check_energy_conditions_full(
        warp_metric, warp_stress, "Static Warp Shell v=0.02"
    )

    # Analyze results
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS - Warp Shell (v=0.02):")
    print("-"*80)

    for condition_name in ['NEC', 'WEC', 'SEC', 'DEC']:
        if condition_name in warp_conditions:
            values = warp_conditions[condition_name]['values']
            min_val = np.min(values)
            max_val = np.max(values)
            violations = np.sum(values < -1e-15)

            print(f"\n{condition_name}:")
            print(f"  Min value: {min_val:.6e}")
            print(f"  Max value: {max_val:.6e}")
            print(f"  Violations (< -1e-15): {violations}")

            if violations > 0:
                print(f"  ⚠ VIOLATIONS DETECTED!")
                print(f"  Number of grid points with violations: {len(np.where(values < -1e-15)[0])}")
                print(f"  Worst violation: {min_val:.6e}")
            else:
                print(f"  ✓ No violations")

    # Summary
    print("\n\n" + "="*80)
    print("CRITICAL FINDINGS SUMMARY")
    print("="*80)
    print()

    # Check v=0 shell
    shell_has_violations = False
    shell_worst = 0
    for condition_name in ['NEC', 'WEC', 'SEC', 'DEC']:
        if condition_name in shell_conditions:
            min_val = np.min(shell_conditions[condition_name]['values'])
            if min_val < -1e-15:
                shell_has_violations = True
                if min_val < shell_worst:
                    shell_worst = min_val

    # Check v=0.02 warp shell
    warp_has_violations = False
    warp_worst = 0
    for condition_name in ['NEC', 'WEC', 'SEC', 'DEC']:
        if condition_name in warp_conditions:
            min_val = np.min(warp_conditions[condition_name]['values'])
            if min_val < -1e-15:
                warp_has_violations = True
                if min_val < warp_worst:
                    warp_worst = min_val

    print(f"Static Shell (v=0):")
    if shell_has_violations:
        print(f"  ⚠ HAS VIOLATIONS!")
        print(f"  Worst violation: {shell_worst:.6e}")
    else:
        print(f"  ✓ No violations detected")

    print()
    print(f"Warp Shell (v=0.02):")
    if warp_has_violations:
        print(f"  ⚠ HAS VIOLATIONS!")
        print(f"  Worst violation: {warp_worst:.6e}")
    else:
        print(f"  ✓ No violations detected")

    print()
    print("="*80)
    print()

    if shell_has_violations or warp_has_violations:
        print("⚠⚠⚠ CRITICAL DISCREPANCY FOUND ⚠⚠⚠")
        print()
        print("The paper reproduction claimed ZERO violations, but this was because")
        print("compute_energy_conditions=False was used in reproduce_results.py line 574.")
        print()
        print("The energy conditions were NEVER ACTUALLY COMPUTED!")
        print()
        print("This explains why acceleration_research3 found violations ~10^40.")
        print("The Fuchs shell DOES have violations even at constant velocity.")
        print()
    else:
        print("✓ Paper claims validated - no violations found")
        print()
        print("If this is the case, the 10^40 violations in acceleration_research3")
        print("must be due to a different issue (time-dependent code, resolution, etc.)")

    print("="*80)

    return {
        'shell_conditions': shell_conditions,
        'warp_conditions': warp_conditions,
        'shell_has_violations': shell_has_violations,
        'warp_has_violations': warp_has_violations,
        'shell_worst': shell_worst,
        'warp_worst': warp_worst
    }


if __name__ == "__main__":
    results = test_static_shells()

    print("\nTest complete. Results stored in 'results' dictionary.")
