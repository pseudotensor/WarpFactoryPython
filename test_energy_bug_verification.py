#!/usr/bin/env python3
"""
Verification test for the inner product bug in energy condition calculations.

This test demonstrates the bug by testing with Minkowski spacetime where:
- Stress-energy tensor is zero everywhere
- All energy conditions should evaluate to exactly zero
- Any non-zero results indicate a bug in the implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from warpfactory.metrics.minkowski.minkowski import get_minkowski_metric
from warpfactory.core.tensor import Tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions


def test_minkowski_energy_conditions():
    """
    Test energy conditions on Minkowski (flat) spacetime.

    For flat spacetime with zero stress-energy:
    - All energy conditions should be exactly zero
    - Any violations indicate implementation bugs
    """
    print("="*80)
    print("MINKOWSKI SPACETIME TEST")
    print("="*80)
    print()
    print("Testing energy conditions on flat spacetime with T_μν = 0")
    print("Expected result: All conditions = 0 (no violations)")
    print()

    # Create small Minkowski metric
    grid_size = [3, 5, 5, 5]
    print(f"Grid size: {grid_size}")

    metric = get_minkowski_metric(grid_size)
    print("✓ Minkowski metric created")

    # Create zero stress-energy tensor
    energy_dict = {}
    for i in range(4):
        for j in range(4):
            energy_dict[(i, j)] = np.zeros(grid_size)

    energy_tensor = Tensor(
        tensor=energy_dict,
        tensor_type="stress-energy",
        name="Zero",
        index="covariant",
        coords="cartesian",
        scaling=[1.0, 1.0, 1.0, 1.0]
    )
    print("✓ Zero stress-energy tensor created")
    print()

    # Test each energy condition
    results = {}

    print("Testing energy conditions:")
    print("-" * 80)

    for condition in ['Null', 'Weak', 'Dominant', 'Strong']:
        print(f"\n{condition} Energy Condition:")

        try:
            # Use fewer test vectors for speed
            condition_map, _, _ = get_energy_conditions(
                energy_tensor,
                metric,
                condition,
                num_angular_vec=20,
                num_time_vec=5
            )

            min_val = np.min(condition_map)
            max_val = np.max(condition_map)
            mean_val = np.mean(condition_map)
            abs_max = np.max(np.abs(condition_map))

            results[condition] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'abs_max': abs_max,
                'passed': abs_max < 1e-10
            }

            print(f"  Min value:     {min_val:.6e}")
            print(f"  Max value:     {max_val:.6e}")
            print(f"  Mean value:    {mean_val:.6e}")
            print(f"  Max |value|:   {abs_max:.6e}")

            if abs_max < 1e-10:
                print(f"  ✓ PASS - Values are essentially zero")
            else:
                print(f"  ✗ FAIL - Non-zero values detected (BUG!)")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[condition] = {'error': str(e)}

    # Summary
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()

    all_passed = True
    bugs_found = []

    for condition in ['Null', 'Weak', 'Dominant', 'Strong']:
        if condition in results:
            if 'error' in results[condition]:
                print(f"✗ {condition:12s}: ERROR - {results[condition]['error']}")
                all_passed = False
                bugs_found.append(condition)
            elif results[condition]['passed']:
                print(f"✓ {condition:12s}: PASS (max |val| = {results[condition]['abs_max']:.2e})")
            else:
                print(f"✗ {condition:12s}: FAIL (max |val| = {results[condition]['abs_max']:.2e})")
                all_passed = False
                bugs_found.append(condition)

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()

    if all_passed:
        print("✓ All energy conditions correctly evaluate to zero on Minkowski spacetime")
        print("  The implementation appears correct.")
    else:
        print("✗ BUG DETECTED: Some energy conditions are non-zero on flat spacetime!")
        print()
        print(f"  Failed conditions: {', '.join(bugs_found)}")
        print()
        print("  This indicates a bug in the implementation.")
        print()

        if 'Dominant' in bugs_found:
            print("  SPECIFIC BUG IDENTIFIED:")
            print("  -------------------------")
            print("  The Dominant Energy Condition uses get_inner_product() which has")
            print("  a bug in the vector contraction when indices are different.")
            print()
            print("  Current (WRONG):")
            print("    for mu in range(4):")
            print("        for nu in range(4):")
            print("            innerprod += vec_a['field'][mu] * vec_b['field'][nu]")
            print()
            print("  Should be (CORRECT):")
            print("    for mu in range(4):")
            print("        innerprod += vec_a['field'][mu] * vec_b['field'][mu]")
            print()
            print("  The double loop creates a 4x4 tensor instead of contracting to a scalar.")
            print("  See ENERGY_CONDITION_BUG_ANALYSIS.md for full details.")

    print()
    print("="*80)

    return all_passed, results


def test_dominant_inner_product_directly():
    """
    Direct test of the inner product bug
    """
    print()
    print("="*80)
    print("DIRECT INNER PRODUCT TEST")
    print("="*80)
    print()
    print("Testing the get_inner_product function directly...")
    print()

    from warpfactory.analyzer.utils import get_inner_product

    # Simple test: vector [1, 0, 0, 0] inner product with itself
    # In Minkowski metric with signature (-,+,+,+):
    # <[1,0,0,0], [1,0,0,0]> = η_μν V^μ V^ν = -1*1^2 = -1

    grid_size = [2, 2, 2, 2]
    metric_mink = get_minkowski_metric(grid_size)

    # Create a simple contravariant vector field: [1, 0, 0, 0] everywhere
    vec = {
        'field': [
            np.ones(grid_size),    # t-component = 1
            np.zeros(grid_size),   # x-component = 0
            np.zeros(grid_size),   # y-component = 0
            np.zeros(grid_size)    # z-component = 0
        ],
        'index': 'contravariant',
        'type': '4-vector'
    }

    # Compute inner product with itself
    result = get_inner_product(vec, vec, metric_mink)

    expected = -1.0  # In Minkowski: <[1,0,0,0], [1,0,0,0]> = -1

    print(f"Test vector: V^μ = [1, 0, 0, 0]")
    print(f"Metric: Minkowski (-,+,+,+)")
    print(f"Expected: <V, V> = -1")
    print(f"Computed: <V, V> = {result[0,0,0,0]:.6e}")
    print()

    if np.abs(result[0,0,0,0] - expected) < 1e-10:
        print("✓ Inner product is correct!")
        return True
    else:
        print(f"✗ Inner product is WRONG!")
        print(f"  Error: {result[0,0,0,0] - expected:.6e}")
        print()
        print("  This confirms the bug in get_inner_product().")
        return False


if __name__ == "__main__":
    print("\n")
    print("*"*80)
    print("*" + " "*78 + "*")
    print("*" + " "*20 + "ENERGY CONDITION BUG VERIFICATION" + " "*26 + "*")
    print("*" + " "*78 + "*")
    print("*"*80)
    print()

    # Test 1: Direct inner product test
    inner_prod_ok = test_dominant_inner_product_directly()

    # Test 2: Full energy conditions on Minkowski
    energy_ok, results = test_minkowski_energy_conditions()

    # Final verdict
    print()
    print("*"*80)
    print("FINAL VERDICT")
    print("*"*80)
    print()

    if energy_ok and inner_prod_ok:
        print("✓ No bugs detected - implementation appears correct")
    else:
        print("✗ BUGS CONFIRMED:")
        if not inner_prod_ok:
            print("  - Inner product contraction is incorrect")
        if not energy_ok:
            print("  - Energy condition calculations have errors")
        print()
        print("See ENERGY_CONDITION_BUG_ANALYSIS.md for detailed analysis and fix.")

    print()
    print("*"*80)
    print()
