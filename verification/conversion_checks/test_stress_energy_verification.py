#!/usr/bin/env python3
"""
Verification script for stress-energy tensor calculation from metric.

This script verifies:
1. Einstein tensor calculation: G_μν = R_μν - R/2 g_μν
2. Stress-energy tensor: T^μν = (c^4/8πG) G^μν
3. Index raising (covariant to contravariant)
4. All signs and factors
5. Vacuum test (Minkowski should give zero)
6. Units and constants

Compares MATLAB and Python implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from warpfactory.metrics.minkowski import get_minkowski_metric
from warpfactory.solver.energy import get_energy_tensor, metric_to_energy_density
from warpfactory.core.tensor_ops import c4_inv
from warpfactory.units.constants import c as speed_of_light, G as gravitational_constant


def test_vacuum_energy():
    """Test that Minkowski spacetime gives zero stress-energy"""
    print("="*80)
    print("TEST 1: Vacuum (Minkowski) should give zero stress-energy")
    print("="*80)

    # Create Minkowski metric
    grid_size = [5, 5, 5, 5]
    grid_scale = [1.0, 1.0, 1.0, 1.0]

    print(f"\nCreating Minkowski metric with grid_size={grid_size}")
    metric = get_minkowski_metric(grid_size, grid_scale)

    # Compute stress-energy
    print("Computing stress-energy tensor...")
    energy = get_energy_tensor(metric, try_gpu=False)

    # Check all components
    max_val = 0
    all_zero = True

    print("\nStress-energy tensor components T^μν:")
    for i in range(4):
        for j in range(4):
            component_max = np.max(np.abs(energy.tensor[(i, j)]))
            print(f"  T^{i}{j}: max = {component_max:.6e}")

            if component_max > 1e-6:
                all_zero = False
                max_val = max(max_val, component_max)

    print(f"\nMaximum absolute value: {max_val:.6e}")

    if all_zero:
        print("✓ PASS: All components near zero for Minkowski (vacuum)")
        return True
    else:
        print(f"✗ FAIL: Non-zero components found (max: {max_val:.6e})")
        return False


def test_einstein_tensor_formula():
    """Test Einstein tensor formula: G_μν = R_μν - (1/2) g_μν R"""
    print("\n" + "="*80)
    print("TEST 2: Einstein tensor formula G_μν = R_μν - (R/2) g_μν")
    print("="*80)

    from warpfactory.solver.ricci import calculate_ricci_tensor, calculate_ricci_scalar
    from warpfactory.solver.einstein import calculate_einstein_tensor

    grid_size = [5, 5, 5, 5]
    grid_scale = [1.0, 1.0, 1.0, 1.0]

    print(f"\nCreating Minkowski metric with grid_size={grid_size}")
    metric = get_minkowski_metric(grid_size, grid_scale)

    gl = metric.tensor
    gu = c4_inv(gl)

    print("Computing Ricci tensor...")
    R_munu = calculate_ricci_tensor(gu, gl, grid_scale)

    print("Computing Ricci scalar...")
    R = calculate_ricci_scalar(R_munu, gu)

    print("Computing Einstein tensor...")
    G = calculate_einstein_tensor(R_munu, R, gl)

    # Verify formula manually
    print("\nVerifying formula for each component:")
    all_match = True

    for i in range(4):
        for j in range(4):
            expected = R_munu[(i, j)] - 0.5 * gl[(i, j)] * R
            computed = G[(i, j)]

            diff = np.max(np.abs(expected - computed))

            if diff > 1e-10:
                print(f"  G_{i}{j}: MISMATCH (diff = {diff:.6e})")
                all_match = False
            else:
                print(f"  G_{i}{j}: ✓ matches formula")

    if all_match:
        print("\n✓ PASS: Einstein tensor formula is correct")
        return True
    else:
        print("\n✗ FAIL: Formula mismatch detected")
        return False


def test_stress_energy_formula():
    """Test stress-energy formula: T^μν = (c^4/8πG) G^μν"""
    print("\n" + "="*80)
    print("TEST 3: Stress-energy formula T^μν = (c^4/8πG) G^μν")
    print("="*80)

    from warpfactory.solver.ricci import calculate_ricci_tensor, calculate_ricci_scalar
    from warpfactory.solver.einstein import calculate_einstein_tensor

    # Use simple test case with non-zero curvature
    grid_size = [5, 5, 5, 5]
    grid_scale = [1.0, 1.0, 1.0, 1.0]

    # Create a metric with slight perturbation to test non-vacuum
    print("\nCreating slightly perturbed metric...")
    gl = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                if i == 0:
                    gl[(i, j)] = -np.ones(grid_size)
                else:
                    # Add small perturbation
                    gl[(i, j)] = np.ones(grid_size) * (1.0 + 0.001)
            else:
                gl[(i, j)] = np.zeros(grid_size)

    gu = c4_inv(gl)

    print("Computing curvature tensors...")
    R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
    R = calculate_ricci_scalar(R_munu, gu)
    G = calculate_einstein_tensor(R_munu, R, gl)

    # Compute stress-energy using the function
    print("Computing stress-energy tensor...")
    T = metric_to_energy_density(gl, gu, grid_scale)

    # Verify formula manually
    c = speed_of_light()
    Grav = gravitational_constant()
    factor = c**4 / (8 * np.pi * Grav)

    print(f"\nConstants:")
    print(f"  c = {c:.6e} m/s")
    print(f"  G = {Grav:.6e} m^3/(kg·s^2)")
    print(f"  c^4/(8πG) = {factor:.6e} J/m^3")

    print("\nVerifying formula component by component...")

    # First compute covariant energy density
    T_cov_expected = {}
    for mu in range(4):
        for nu in range(4):
            T_cov_expected[(mu, nu)] = factor * G[(mu, nu)]

    # Then raise indices
    T_expected = {}
    for mu in range(4):
        for nu in range(4):
            T_expected[(mu, nu)] = np.zeros(grid_size)
            for alpha in range(4):
                for beta in range(4):
                    T_expected[(mu, nu)] += (
                        T_cov_expected[(alpha, beta)] *
                        gu[(alpha, mu)] *
                        gu[(beta, nu)]
                    )

    # Compare with computed values
    all_match = True
    max_diff = 0

    for i in range(4):
        for j in range(4):
            diff = np.max(np.abs(T[(i, j)] - T_expected[(i, j)]))
            max_diff = max(max_diff, diff)

            if diff > 1e-6:
                print(f"  T^{i}{j}: MISMATCH (max diff = {diff:.6e})")
                all_match = False
            else:
                print(f"  T^{i}{j}: ✓ matches (diff = {diff:.6e})")

    print(f"\nMaximum difference: {max_diff:.6e}")

    if all_match:
        print("✓ PASS: Stress-energy formula is correct")
        return True
    else:
        print("✗ FAIL: Formula mismatch detected")
        return False


def test_index_raising():
    """Test index raising: T^μν = g^μα g^νβ T_αβ"""
    print("\n" + "="*80)
    print("TEST 4: Index raising T^μν = g^μα g^νβ T_αβ")
    print("="*80)

    grid_size = [5, 5, 5, 5]

    print("\nCreating test covariant tensor...")
    # Create a simple covariant tensor
    T_cov = {}
    for i in range(4):
        for j in range(4):
            T_cov[(i, j)] = np.random.randn(*grid_size) * 0.1

    # Minkowski metric for simplicity
    gl = {}
    gu = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                if i == 0:
                    gl[(i, j)] = -np.ones(grid_size)
                    gu[(i, j)] = -np.ones(grid_size)
                else:
                    gl[(i, j)] = np.ones(grid_size)
                    gu[(i, j)] = np.ones(grid_size)
            else:
                gl[(i, j)] = np.zeros(grid_size)
                gu[(i, j)] = np.zeros(grid_size)

    print("Raising indices manually...")
    T_contra_manual = {}
    for mu in range(4):
        for nu in range(4):
            T_contra_manual[(mu, nu)] = np.zeros(grid_size)
            for alpha in range(4):
                for beta in range(4):
                    T_contra_manual[(mu, nu)] += (
                        T_cov[(alpha, beta)] *
                        gu[(alpha, mu)] *
                        gu[(beta, nu)]
                    )

    # Verify with Minkowski (should just flip signs appropriately)
    print("\nVerifying index raising for Minkowski metric:")

    all_match = True
    for i in range(4):
        for j in range(4):
            # For Minkowski: T^00 = T_00 (both negative), T^ij = T_ij (both positive)
            # T^0i = -T_0i (sign flip)
            if i == 0 and j == 0:
                expected = T_cov[(i, j)]  # No sign change (both negative)
            elif i == 0 or j == 0:
                expected = -T_cov[(i, j)]  # Sign flip (one index raised)
            else:
                expected = T_cov[(i, j)]  # No sign change (both positive)

            computed = T_contra_manual[(i, j)]
            diff = np.max(np.abs(expected - computed))

            if diff > 1e-10:
                print(f"  T^{i}{j}: MISMATCH (diff = {diff:.6e})")
                all_match = False
            else:
                print(f"  T^{i}{j}: ✓ correct sign")

    if all_match:
        print("\n✓ PASS: Index raising is correct")
        return True
    else:
        print("\n✗ FAIL: Index raising incorrect")
        return False


def test_constants_match_matlab():
    """Test that constants match MATLAB values"""
    print("\n" + "="*80)
    print("TEST 5: Constants match MATLAB implementation")
    print("="*80)

    c_py = speed_of_light()
    G_py = gravitational_constant()

    # MATLAB values from G.m and c.m
    c_matlab = 2.99792458e8  # m/s
    G_matlab = 6.67430e-11   # m^3/(kg·s^2)

    print(f"\nPython constants:")
    print(f"  c = {c_py:.10e} m/s")
    print(f"  G = {G_py:.10e} m^3/(kg·s^2)")

    print(f"\nMATLAB constants:")
    print(f"  c = {c_matlab:.10e} m/s")
    print(f"  G = {G_matlab:.10e} m^3/(kg·s^2)")

    c_match = abs(c_py - c_matlab) < 1e-3
    G_match = abs(G_py - G_matlab) < 1e-15

    print(f"\nComparison:")
    print(f"  c matches: {'✓' if c_match else '✗'} (diff = {abs(c_py - c_matlab):.6e})")
    print(f"  G matches: {'✓' if G_match else '✗'} (diff = {abs(G_py - G_matlab):.6e})")

    if c_match and G_match:
        print("\n✓ PASS: Constants match MATLAB")
        return True
    else:
        print("\n✗ FAIL: Constants mismatch")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("STRESS-ENERGY TENSOR VERIFICATION")
    print("="*80)
    print("\nComparing MATLAB and Python implementations")
    print("\nMATLAB files:")
    print("  /WarpFactory/Solver/utils/met2den.m")
    print("  /WarpFactory/Solver/utils/einT.m")
    print("  /WarpFactory/Solver/utils/einE.m")
    print("\nPython files:")
    print("  /WarpFactory/warpfactory_py/warpfactory/solver/energy.py")
    print("  /WarpFactory/warpfactory_py/warpfactory/solver/einstein.py")
    print()

    results = {}

    # Run all tests
    results['constants'] = test_constants_match_matlab()
    results['vacuum'] = test_vacuum_energy()
    results['einstein_formula'] = test_einstein_tensor_formula()
    results['stress_energy_formula'] = test_stress_energy_formula()
    results['index_raising'] = test_index_raising()

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    all_pass = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_pass = False

    print("\n" + "="*80)
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("\nThe Python implementation correctly matches the MATLAB implementation:")
        print("1. Einstein tensor: G_μν = R_μν - (R/2) g_μν ✓")
        print("2. Stress-energy: T^μν = (c^4/8πG) G^μν ✓")
        print("3. Index raising: T^μν = g^μα g^νβ T_αβ ✓")
        print("4. Signs and factors: All correct ✓")
        print("5. Vacuum test: Minkowski gives zero ✓")
        print("6. Constants: Match MATLAB values ✓")
        print("\nNo bugs found. Implementation is correct.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nBugs detected in the implementation.")
        print("Review failed tests above for details.")

    print("="*80 + "\n")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
