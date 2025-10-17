#!/usr/bin/env python3
"""
Verify stress-energy calculation against Schwarzschild metric (vacuum solution).

The Schwarzschild metric is an exact vacuum solution to Einstein's field equations,
so the stress-energy tensor should be exactly zero everywhere (except at r=0 singularity).

This provides a strong validation that the entire pipeline is correct:
metric → Ricci tensor → Ricci scalar → Einstein tensor → stress-energy tensor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from warpfactory.metrics.schwarzschild import get_schwarzschild_metric
from warpfactory.solver.energy import get_energy_tensor


def test_schwarzschild_vacuum():
    """
    Test that Schwarzschild metric gives zero stress-energy (vacuum solution).

    The Schwarzschild metric in Schwarzschild coordinates:
    ds² = -(1 - 2M/r)dt² + (1 - 2M/r)⁻¹dr² + r²dθ² + r²sin²θ dφ²

    This is an EXACT vacuum solution, so T^μν = 0 everywhere (except at r=0).
    """
    print("="*80)
    print("SCHWARZSCHILD VACUUM SOLUTION TEST")
    print("="*80)
    print("\nTesting that Schwarzschild metric gives zero stress-energy")
    print("(vacuum solution to Einstein's field equations)")

    # Create Schwarzschild metric
    # Use a region away from r=0 singularity and event horizon
    M = 1.0  # Mass (in geometric units where G=c=1)
    r_min = 10.0 * M  # Well outside event horizon (2M)
    r_max = 20.0 * M

    grid_size = [5, 7, 7, 7]  # [t, r, theta, phi]

    print(f"\nParameters:")
    print(f"  Mass M = {M}")
    print(f"  Radial range: {r_min} to {r_max} (event horizon at r=2M={2*M})")
    print(f"  Grid size: {grid_size}")

    print("\nCreating Schwarzschild metric...")

    try:
        metric = get_schwarzschild_metric(
            grid_size=grid_size,
            M=M,
            r_min=r_min,
            r_max=r_max
        )
        print("✓ Metric created")
    except Exception as e:
        print(f"✗ Failed to create metric: {e}")
        return False

    # Compute stress-energy
    print("\nComputing stress-energy tensor...")
    print("(This should be zero for a vacuum solution)")

    try:
        energy = get_energy_tensor(metric, try_gpu=False)
        print("✓ Stress-energy computed")
    except Exception as e:
        print(f"✗ Failed to compute stress-energy: {e}")
        return False

    # Check all components
    print("\nStress-energy tensor components T^μν:")

    max_val = 0
    all_near_zero = True
    tolerance = 1e-6  # More lenient due to numerical derivatives

    for i in range(4):
        for j in range(4):
            component_max = np.max(np.abs(energy.tensor[(i, j)]))
            component_mean = np.mean(np.abs(energy.tensor[(i, j)]))

            status = "✓" if component_max < tolerance else "✗"
            print(f"  {status} T^{i}{j}: max = {component_max:.6e}, mean = {component_mean:.6e}")

            if component_max > tolerance:
                all_near_zero = False

            max_val = max(max_val, component_max)

    print(f"\nMaximum absolute value: {max_val:.6e}")
    print(f"Tolerance: {tolerance:.6e}")

    # Verdict
    print("\n" + "="*80)
    if all_near_zero:
        print("✓ PASS: Schwarzschild metric gives zero stress-energy")
        print("\nThis confirms the entire pipeline is working correctly:")
        print("  1. Metric tensor correctly defined")
        print("  2. Ricci tensor correctly computed")
        print("  3. Ricci scalar correctly computed")
        print("  4. Einstein tensor correctly computed")
        print("  5. Stress-energy tensor correctly computed")
        print("  6. Known vacuum solution verified")
        return True
    else:
        print("✗ FAIL: Non-zero stress-energy found")
        print(f"\nExpected: All components < {tolerance:.6e}")
        print(f"Found: Maximum component = {max_val:.6e}")
        print("\nPossible causes:")
        print("  1. Error in metric definition")
        print("  2. Error in Ricci tensor computation")
        print("  3. Error in Einstein tensor computation")
        print("  4. Error in stress-energy computation")
        print("  5. Numerical errors from finite differences")
        return False


def test_minkowski_limit():
    """
    Test that Schwarzschild metric approaches Minkowski as M→0 or r→∞.
    """
    print("\n" + "="*80)
    print("SCHWARZSCHILD → MINKOWSKI LIMIT TEST")
    print("="*80)
    print("\nTesting that Schwarzschild approaches flat space as M→0")

    # Create Schwarzschild with very small mass
    M = 1e-10
    r_min = 10.0
    r_max = 20.0
    grid_size = [5, 7, 7, 7]

    print(f"\nParameters:")
    print(f"  Mass M = {M} (very small)")
    print(f"  Radial range: {r_min} to {r_max}")
    print(f"  Expected: Metric ≈ Minkowski, T^μν ≈ 0")

    try:
        metric = get_schwarzschild_metric(
            grid_size=grid_size,
            M=M,
            r_min=r_min,
            r_max=r_max
        )

        energy = get_energy_tensor(metric, try_gpu=False)

        max_val = max(
            np.max(np.abs(energy.tensor[(i, j)]))
            for i in range(4) for j in range(4)
        )

        print(f"\nMaximum |T^μν|: {max_val:.6e}")

        if max_val < 1e-6:
            print("✓ PASS: Minkowski limit verified")
            return True
        else:
            print("✗ FAIL: Non-zero stress-energy in flat space limit")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all Schwarzschild verification tests"""

    print("\n" + "="*80)
    print("SCHWARZSCHILD METRIC VERIFICATION")
    print("="*80)
    print("\nValidating stress-energy calculation with exact vacuum solution")
    print()

    results = {}

    # Test 1: Schwarzschild is vacuum
    try:
        results['schwarzschild_vacuum'] = test_schwarzschild_vacuum()
    except Exception as e:
        print(f"\n✗ EXCEPTION in schwarzschild_vacuum: {e}")
        results['schwarzschild_vacuum'] = False

    # Test 2: Minkowski limit
    try:
        results['minkowski_limit'] = test_minkowski_limit()
    except Exception as e:
        print(f"\n✗ EXCEPTION in minkowski_limit: {e}")
        results['minkowski_limit'] = False

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_pass = all(results.values())

    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("\nThe stress-energy calculation is validated against")
        print("the Schwarzschild metric (exact vacuum solution).")
        print("\nThis confirms the implementation is correct.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nReview the failed tests above.")
    print("="*80 + "\n")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
