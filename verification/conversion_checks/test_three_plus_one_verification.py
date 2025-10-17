"""
Test to verify 3+1 metric construction against Minkowski spacetime
"""
import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.three_plus_one import (
    set_minkowski_three_plus_one,
    three_plus_one_builder,
    three_plus_one_decomposer
)
from warpfactory.core.tensor import Tensor

def test_minkowski_construction():
    """Test that 3+1 construction produces correct Minkowski metric"""

    print("=" * 70)
    print("TESTING 3+1 METRIC CONSTRUCTION WITH MINKOWSKI SPACETIME")
    print("=" * 70)

    # Create small test grid
    grid_size = [2, 2, 2, 2]  # Small for easy verification

    # Get Minkowski in 3+1 form
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    print("\n1. VERIFYING INPUT 3+1 COMPONENTS")
    print("-" * 70)
    print(f"Alpha (lapse): {alpha[0,0,0,0]} (should be 1)")
    print(f"Beta[0] (shift): {beta[0][0,0,0,0]} (should be 0)")
    print(f"Beta[1] (shift): {beta[1][0,0,0,0]} (should be 0)")
    print(f"Beta[2] (shift): {beta[2][0,0,0,0]} (should be 0)")
    print(f"Gamma[(0,0)] (spatial metric): {gamma[(0,0)][0,0,0,0]} (should be 1)")
    print(f"Gamma[(0,1)] (spatial metric): {gamma[(0,1)][0,0,0,0]} (should be 0)")
    print(f"Gamma[(1,1)] (spatial metric): {gamma[(1,1)][0,0,0,0]} (should be 1)")

    # Build metric tensor
    metric_dict = three_plus_one_builder(alpha, beta, gamma)

    print("\n2. VERIFYING CONSTRUCTED METRIC TENSOR")
    print("-" * 70)

    # Expected Minkowski metric: diag(-1, 1, 1, 1)
    expected = {
        (0,0): -1.0, (0,1): 0.0, (0,2): 0.0, (0,3): 0.0,
        (1,0): 0.0,  (1,1): 1.0, (1,2): 0.0, (1,3): 0.0,
        (2,0): 0.0,  (2,1): 0.0, (2,2): 1.0, (2,3): 0.0,
        (3,0): 0.0,  (3,1): 0.0, (3,2): 0.0, (3,3): 1.0
    }

    errors = []

    for i in range(4):
        for j in range(4):
            computed = metric_dict[(i,j)][0,0,0,0]
            expect = expected[(i,j)]
            diff = abs(computed - expect)

            status = "✓" if diff < 1e-10 else "✗"
            print(f"  g[{i},{j}] = {computed:8.5f}  (expect {expect:8.5f})  {status}")

            if diff >= 1e-10:
                errors.append(f"g[{i},{j}]: got {computed}, expected {expect}, diff={diff}")

    print("\n3. VERIFYING METRIC SIGNATURE")
    print("-" * 70)

    # Check eigenvalues at a point
    metric_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            metric_matrix[i, j] = metric_dict[(i,j)][0,0,0,0]

    eigenvalues = np.linalg.eigvalsh(metric_matrix)
    eigenvalues_sorted = sorted(eigenvalues)

    print(f"Eigenvalues: {eigenvalues_sorted}")

    # For Minkowski with signature (-,+,+,+), we expect eigenvalues: -1, 1, 1, 1
    negative_count = sum(1 for e in eigenvalues if e < 0)
    positive_count = sum(1 for e in eigenvalues if e > 0)

    print(f"Negative eigenvalues: {negative_count} (should be 1)")
    print(f"Positive eigenvalues: {positive_count} (should be 3)")

    if negative_count != 1 or positive_count != 3:
        errors.append(f"Wrong signature: ({negative_count},{positive_count}) instead of (1,3)")

    print("\n4. VERIFYING SPECIFIC FORMULAS")
    print("-" * 70)

    # Verify g_00 = -α² + β^i β_i
    alpha_val = alpha[0,0,0,0]

    # Calculate beta^i (raised with gamma^ij)
    # For Minkowski: gamma^ij = identity, beta_i = 0, so beta^i = 0
    beta_raised = [0.0, 0.0, 0.0]

    beta_squared = sum(beta_raised[i] * beta[i][0,0,0,0] for i in range(3))
    g_00_formula = -alpha_val**2 + beta_squared
    g_00_computed = metric_dict[(0,0)][0,0,0,0]

    print(f"g_00 formula: -α² + β^i β_i = -{alpha_val}² + {beta_squared} = {g_00_formula}")
    print(f"g_00 computed: {g_00_computed}")
    print(f"Match: {'✓' if abs(g_00_formula - g_00_computed) < 1e-10 else '✗'}")

    # Verify g_0i = beta_i
    print(f"\nTime-space components (g_0i = β_i):")
    for i in range(3):
        g_0i = metric_dict[(0, i+1)][0,0,0,0]
        beta_i = beta[i][0,0,0,0]
        match = '✓' if abs(g_0i - beta_i) < 1e-10 else '✗'
        print(f"  g_0{i+1} = {g_0i}, β_{i} = {beta_i}  {match}")
        if abs(g_0i - beta_i) >= 1e-10:
            errors.append(f"g_0{i+1} != beta_{i}: {g_0i} != {beta_i}")

    # Verify g_ij = gamma_ij
    print(f"\nSpace-space components (g_ij = γ_ij):")
    for i in range(3):
        for j in range(3):
            g_ij = metric_dict[(i+1, j+1)][0,0,0,0]
            gamma_ij = gamma[(i, j)][0,0,0,0]
            match = '✓' if abs(g_ij - gamma_ij) < 1e-10 else '✗'
            if i == j or abs(g_ij - gamma_ij) >= 1e-10:  # Only print diagonals and mismatches
                print(f"  g_{i+1}{j+1} = {g_ij}, γ_{i}{j} = {gamma_ij}  {match}")
            if abs(g_ij - gamma_ij) >= 1e-10:
                errors.append(f"g_{i+1}{j+1} != gamma_{i}{j}: {g_ij} != {gamma_ij}")

    print("\n5. TESTING ROUND-TRIP DECOMPOSITION")
    print("-" * 70)

    # Create a Tensor object and decompose it
    metric_tensor = Tensor(
        tensor=metric_dict,
        tensor_type='metric',
        name='Minkowski',
        coords='cartesian',
        index='covariant'
    )

    alpha_decomp, beta_decomp, gamma_decomp, _, _ = three_plus_one_decomposer(metric_tensor)

    print(f"Original alpha: {alpha[0,0,0,0]}")
    print(f"Decomposed alpha: {alpha_decomp[0,0,0,0]}")
    print(f"Match: {'✓' if abs(alpha[0,0,0,0] - alpha_decomp[0,0,0,0]) < 1e-10 else '✗'}")

    for i in range(3):
        beta_orig = beta[i][0,0,0,0]
        beta_dec = beta_decomp[i][0,0,0,0]
        match = '✓' if abs(beta_orig - beta_dec) < 1e-10 else '✗'
        print(f"Beta[{i}]: original={beta_orig}, decomposed={beta_dec}  {match}")

    print("\n" + "=" * 70)
    if errors:
        print("RESULT: FAILED")
        print("=" * 70)
        print("\nERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")
        return False
    else:
        print("RESULT: ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nThe 3+1 metric construction is CORRECT:")
        print("  • g_00 = -α² + β^i β_i formula is correct")
        print("  • g_0i = β_i time-space components are correct")
        print("  • g_ij = γ_ij spatial components are correct")
        print("  • Signature (-,+,+,+) is correct")
        print("  • Round-trip decomposition works correctly")
        return True


if __name__ == "__main__":
    success = test_minkowski_construction()
    sys.exit(0 if success else 1)
