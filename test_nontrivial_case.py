"""
Test 3+1 construction with non-trivial case (non-zero beta)
"""
import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.three_plus_one import three_plus_one_builder

def test_nontrivial_metric():
    """Test with non-zero shift vector to verify beta^i beta_i term"""

    print("=" * 70)
    print("TESTING 3+1 CONSTRUCTION WITH NON-TRIVIAL METRIC")
    print("=" * 70)

    # Create simple test case with non-zero shift
    grid_size = [1, 1, 1, 1]

    # Set up 3+1 components
    alpha = np.ones(grid_size) * 2.0  # Lapse = 2

    # Non-zero shift vector (covariant)
    beta = {
        0: np.ones(grid_size) * 0.3,  # beta_x = 0.3
        1: np.ones(grid_size) * 0.4,  # beta_y = 0.4
        2: np.zeros(grid_size)         # beta_z = 0
    }

    # Non-trivial spatial metric
    gamma = {
        (0, 0): np.ones(grid_size) * 1.5,
        (0, 1): np.ones(grid_size) * 0.1,
        (0, 2): np.zeros(grid_size),
        (1, 0): np.ones(grid_size) * 0.1,
        (1, 1): np.ones(grid_size) * 2.0,
        (1, 2): np.zeros(grid_size),
        (2, 0): np.zeros(grid_size),
        (2, 1): np.zeros(grid_size),
        (2, 2): np.ones(grid_size) * 1.0,
    }

    print("\nInput 3+1 components:")
    print("-" * 70)
    print(f"α = {alpha[0,0,0,0]}")
    print(f"β_0 = {beta[0][0,0,0,0]}")
    print(f"β_1 = {beta[1][0,0,0,0]}")
    print(f"β_2 = {beta[2][0,0,0,0]}")
    print("\nγ_ij (spatial metric):")
    for i in range(3):
        row = [gamma[(i,j)][0,0,0,0] for j in range(3)]
        print(f"  [{row[0]:6.3f}, {row[1]:6.3f}, {row[2]:6.3f}]")

    # Build metric
    metric = three_plus_one_builder(alpha, beta, gamma)

    # Manual calculation for verification
    print("\n" + "=" * 70)
    print("MANUAL CALCULATION OF g_00 = -α² + β^i β_i")
    print("=" * 70)

    # First compute gamma^ij (inverse of gamma_ij)
    gamma_matrix = np.array([
        [gamma[(0,0)][0,0,0,0], gamma[(0,1)][0,0,0,0], gamma[(0,2)][0,0,0,0]],
        [gamma[(1,0)][0,0,0,0], gamma[(1,1)][0,0,0,0], gamma[(1,2)][0,0,0,0]],
        [gamma[(2,0)][0,0,0,0], gamma[(2,1)][0,0,0,0], gamma[(2,2)][0,0,0,0]],
    ])

    gamma_inv = np.linalg.inv(gamma_matrix)

    print("\nγ^ij (inverse spatial metric):")
    for i in range(3):
        print(f"  [{gamma_inv[i,0]:8.5f}, {gamma_inv[i,1]:8.5f}, {gamma_inv[i,2]:8.5f}]")

    # Compute beta^i = gamma^ij * beta_j
    beta_down = np.array([beta[0][0,0,0,0], beta[1][0,0,0,0], beta[2][0,0,0,0]])
    beta_up = gamma_inv @ beta_down

    print(f"\nβ^i (contravariant shift):")
    print(f"  β^0 = {beta_up[0]:.6f}")
    print(f"  β^1 = {beta_up[1]:.6f}")
    print(f"  β^2 = {beta_up[2]:.6f}")

    # Compute beta^i * beta_i
    beta_squared = np.dot(beta_up, beta_down)

    print(f"\nβ^i β_i = {beta_squared:.6f}")

    # Compute g_00
    alpha_val = alpha[0,0,0,0]
    g_00_manual = -alpha_val**2 + beta_squared

    print(f"\ng_00 = -α² + β^i β_i")
    print(f"     = -{alpha_val}² + {beta_squared:.6f}")
    print(f"     = {g_00_manual:.6f}")

    g_00_computed = metric[(0,0)][0,0,0,0]
    print(f"\ng_00 from three_plus_one_builder: {g_00_computed:.6f}")

    diff = abs(g_00_manual - g_00_computed)
    print(f"Difference: {diff:.2e}")

    success = diff < 1e-10

    if success:
        print("\n✓ g_00 MATCHES MANUAL CALCULATION")
    else:
        print("\n✗ g_00 DOES NOT MATCH")
        return False

    print("\n" + "=" * 70)
    print("VERIFYING TIME-SPACE COMPONENTS")
    print("=" * 70)

    errors = []
    for i in range(3):
        g_0i = metric[(0, i+1)][0,0,0,0]
        beta_i = beta[i][0,0,0,0]
        g_i0 = metric[(i+1, 0)][0,0,0,0]

        print(f"g_0{i+1} = {g_0i:.6f}, β_{i} = {beta_i:.6f}, diff = {abs(g_0i - beta_i):.2e}")

        if abs(g_0i - beta_i) >= 1e-10:
            errors.append(f"g_0{i+1} != β_{i}")

        if abs(g_i0 - beta_i) >= 1e-10:
            errors.append(f"g_{i+1}0 != β_{i} (symmetry broken)")

        if abs(g_0i - g_i0) >= 1e-10:
            errors.append(f"g_0{i+1} != g_{i+1}0 (not symmetric)")

    if not errors:
        print("✓ All time-space components correct")
    else:
        print("✗ Errors in time-space components:")
        for e in errors:
            print(f"  {e}")
        return False

    print("\n" + "=" * 70)
    print("VERIFYING SPACE-SPACE COMPONENTS")
    print("=" * 70)

    for i in range(3):
        for j in range(3):
            g_ij = metric[(i+1, j+1)][0,0,0,0]
            gamma_ij = gamma[(i, j)][0,0,0,0]

            if abs(g_ij - gamma_ij) >= 1e-10:
                print(f"✗ g_{i+1}{j+1} != γ_{i}{j}: {g_ij} != {gamma_ij}")
                return False

    print("✓ All space-space components correct (g_ij = γ_ij)")

    print("\n" + "=" * 70)
    print("VERIFYING METRIC SIGNATURE")
    print("=" * 70)

    # Check signature at a point
    metric_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            metric_matrix[i, j] = metric[(i,j)][0,0,0,0]

    eigenvalues = sorted(np.linalg.eigvalsh(metric_matrix))

    negative_count = sum(1 for e in eigenvalues if e < -1e-10)
    positive_count = sum(1 for e in eigenvalues if e > 1e-10)

    print(f"Eigenvalues: {[f'{e:.6f}' for e in eigenvalues]}")
    print(f"Negative: {negative_count}, Positive: {positive_count}")

    if negative_count == 1 and positive_count == 3:
        print("✓ Signature is (-,+,+,+)")
    else:
        print(f"✗ Wrong signature: ({negative_count},{positive_count})")
        return False

    # Check that metric is actually Lorentzian (g_00 < 0 for this case)
    if g_00_computed < 0:
        print(f"✓ g_00 = {g_00_computed:.6f} < 0 (timelike signature)")
    else:
        print(f"✗ g_00 = {g_00_computed:.6f} >= 0 (wrong signature)")
        return False

    print("\n" + "=" * 70)
    print("RESULT: ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nVerified with non-trivial metric:")
    print("  • Non-zero shift vector (β ≠ 0)")
    print("  • Non-diagonal spatial metric (γ_01 ≠ 0)")
    print("  • g_00 formula correct: -α² + β^i β_i")
    print("  • Beta raising correct: β^i = γ^ij β_j")
    print("  • Signature (-,+,+,+) maintained")

    return True


if __name__ == "__main__":
    success = test_nontrivial_metric()
    sys.exit(0 if success else 1)
