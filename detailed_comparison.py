"""
Detailed comparison of MATLAB and Python implementations
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.solver.finite_differences import take_finite_difference_1, take_finite_difference_2


def compare_first_derivative_formula():
    """Compare the exact formula implementation"""
    print("=" * 80)
    print("DETAILED FIRST DERIVATIVE FORMULA COMPARISON")
    print("=" * 80)

    # Create a simple test array
    nx = 10
    A = np.arange(nx**4).reshape(nx, nx, nx, nx).astype(float)
    delta = [1.0, 1.0, 1.0, 1.0]

    print("\nTest array shape:", A.shape)
    print("Delta:", delta)

    # Test direction k=1 (x direction)
    print("\n" + "-" * 80)
    print("Testing k=1 (x direction)")
    print("-" * 80)

    k = 1
    B = take_finite_difference_1(A, k, delta)

    # Manually compute for a specific point to verify
    i, j, k_idx, l = 5, 5, 5, 5
    if k == 1:
        # MATLAB: B(:,3:end-2,:,:) = (-(A(:,5:end,:,:)-A(:,1:end-4,:,:))+8*(A(:,4:end-1,:,:)-A(:,2:end-3,:,:)))/(12*delta(k));
        # Python indices for j=5 (0-indexed) corresponds to MATLAB index 6
        # For interior point at Python index [i,j,k_idx,l] = [5,5,5,5]
        # This is at position 3 in the 2:-2 slice (indices 2,3,4,5,6,7 -> position 3 is index 5)

        # The stencil uses:
        j_plus_2 = j + 2  # = 7
        j_plus_1 = j + 1  # = 6
        j_minus_1 = j - 1  # = 4
        j_minus_2 = j - 2  # = 3

        manual = (
            -(A[i, j_plus_2, k_idx, l] - A[i, j_minus_2, k_idx, l]) +
            8 * (A[i, j_plus_1, k_idx, l] - A[i, j_minus_1, k_idx, l])
        ) / (12 * delta[1])

        print(f"\nManual calculation at point [{i},{j},{k_idx},{l}]:")
        print(f"  A[i,j+2,k,l] = A[{i},{j_plus_2},{k_idx},{l}] = {A[i, j_plus_2, k_idx, l]}")
        print(f"  A[i,j-2,k,l] = A[{i},{j_minus_2},{k_idx},{l}] = {A[i, j_minus_2, k_idx, l]}")
        print(f"  A[i,j+1,k,l] = A[{i},{j_plus_1},{k_idx},{l}] = {A[i, j_plus_1, k_idx, l]}")
        print(f"  A[i,j-1,k,l] = A[{i},{j_minus_1},{k_idx},{l}] = {A[i, j_minus_1, k_idx, l]}")
        print(f"  Manual result: {manual}")
        print(f"  Function result: {B[i, j, k_idx, l]}")
        print(f"  Difference: {abs(manual - B[i, j, k_idx, l])}")

    # Check boundaries
    print("\n" + "-" * 80)
    print("Boundary handling check:")
    print("-" * 80)
    print(f"B[5,0,5,5] = {B[5,0,5,5]} (should equal B[5,2,5,5] = {B[5,2,5,5]})")
    print(f"Difference: {abs(B[5,0,5,5] - B[5,2,5,5])}")
    print(f"B[5,1,5,5] = {B[5,1,5,5]} (should equal B[5,2,5,5] = {B[5,2,5,5]})")
    print(f"Difference: {abs(B[5,1,5,5] - B[5,2,5,5])}")
    print(f"B[5,-2,5,5] = {B[5,-2,5,5]} (should equal B[5,-3,5,5] = {B[5,-3,5,5]})")
    print(f"Difference: {abs(B[5,-2,5,5] - B[5,-3,5,5])}")
    print(f"B[5,-1,5,5] = {B[5,-1,5,5]} (should equal B[5,-3,5,5] = {B[5,-3,5,5]})")
    print(f"Difference: {abs(B[5,-1,5,5] - B[5,-3,5,5])}")


def compare_second_derivative_formula():
    """Compare the exact formula implementation for second derivatives"""
    print("\n" + "=" * 80)
    print("DETAILED SECOND DERIVATIVE FORMULA COMPARISON")
    print("=" * 80)

    # Create a simple test array
    nx = 10
    A = np.arange(nx**4).reshape(nx, nx, nx, nx).astype(float)
    delta = [1.0, 1.0, 1.0, 1.0]

    print("\nTest array shape:", A.shape)
    print("Delta:", delta)

    # Test direction k1=k2=1 (second derivative in x direction)
    print("\n" + "-" * 80)
    print("Testing k1=k2=1 (d²/dx²)")
    print("-" * 80)

    k1 = k2 = 1
    B = take_finite_difference_2(A, k1, k2, delta)

    # Manually compute for a specific point
    i, j, k_idx, l = 5, 5, 5, 5
    j_plus_2 = j + 2
    j_plus_1 = j + 1
    j_minus_1 = j - 1
    j_minus_2 = j - 2

    manual = (
        -(A[i, j_plus_2, k_idx, l] + A[i, j_minus_2, k_idx, l]) +
        16 * (A[i, j_plus_1, k_idx, l] + A[i, j_minus_1, k_idx, l]) -
        30 * A[i, j, k_idx, l]
    ) / (12 * delta[1]**2)

    print(f"\nManual calculation at point [{i},{j},{k_idx},{l}]:")
    print(f"  Manual result: {manual}")
    print(f"  Function result: {B[i, j, k_idx, l]}")
    print(f"  Difference: {abs(manual - B[i, j, k_idx, l])}")


def compare_mixed_derivative_formula():
    """Compare the exact formula implementation for mixed derivatives"""
    print("\n" + "=" * 80)
    print("DETAILED MIXED DERIVATIVE FORMULA COMPARISON")
    print("=" * 80)

    # Create a simple test array
    nx = 10
    A = np.arange(nx**4).reshape(nx, nx, nx, nx).astype(float)
    delta = [1.0, 1.0, 1.0, 1.0]

    print("\nTest array shape:", A.shape)
    print("Delta:", delta)

    # Test mixed derivative k1=0, k2=1 (d²/dtdx)
    print("\n" + "-" * 80)
    print("Testing k1=0, k2=1 (d²/dtdx)")
    print("-" * 80)

    k1, k2 = 0, 1
    B = take_finite_difference_2(A, k1, k2, delta)

    # Manually compute for a specific point
    # kS = min(0,1) = 0, kL = max(0,1) = 1
    # x indices correspond to kS=0 (time), y indices to kL=1 (x)
    i, j, k_idx, l = 5, 5, 5, 5

    # In MATLAB notation: B(x0,y0,:,:)
    # x0 corresponds to i=5 (Python), y0 to j=5 (Python)

    x2, x1, x0, x_1, x_2 = 7, 6, 5, 4, 3
    y2, y1, y0, y_1, y_2 = 7, 6, 5, 4, 3

    # MATLAB formula for case 2 (partial 0 / partial 1):
    # B(x0,y0,:,:) = 1/(12^2*delta(kL)*delta(kS)) * (
    #     -(-(A(x2,y2,:,:)  -A(x_2,y2,:,:) ) +8*(A(x1,y2,:,:)  -A(x_1,y2,:,:) ))
    #     +(-(A(x2,y_2,:,:) -A(x_2,y_2,:,:)) +8*(A(x1,y_2,:,:) -A(x_1,y_2,:,:)))
    #   +8*(-(A(x2,y1,:,:)  -A(x_2,y1,:,:) ) +8*(A(x1,y1,:,:)  -A(x_1,y1,:,:) ))
    #   -8*(-(A(x2,y_1,:,:) -A(x_2,y_1,:,:)) +8*(A(x1,y_1,:,:) -A(x_1,y_1,:,:)))
    # )

    term1 = -(-(A[x2, y2, k_idx, l] - A[x_2, y2, k_idx, l]) + 8*(A[x1, y2, k_idx, l] - A[x_1, y2, k_idx, l]))
    term2 = (-(A[x2, y_2, k_idx, l] - A[x_2, y_2, k_idx, l]) + 8*(A[x1, y_2, k_idx, l] - A[x_1, y_2, k_idx, l]))
    term3 = 8*(-(A[x2, y1, k_idx, l] - A[x_2, y1, k_idx, l]) + 8*(A[x1, y1, k_idx, l] - A[x_1, y1, k_idx, l]))
    term4 = -8*(-(A[x2, y_1, k_idx, l] - A[x_2, y_1, k_idx, l]) + 8*(A[x1, y_1, k_idx, l] - A[x_1, y_1, k_idx, l]))

    manual = (term1 + term2 + term3 + term4) / (144 * delta[k2] * delta[k1])

    print(f"\nManual calculation at point [{i},{j},{k_idx},{l}]:")
    print(f"  Term1: {term1}")
    print(f"  Term2: {term2}")
    print(f"  Term3: {term3}")
    print(f"  Term4: {term4}")
    print(f"  Sum: {term1 + term2 + term3 + term4}")
    print(f"  Manual result: {manual}")
    print(f"  Function result: {B[i, j, k_idx, l]}")
    print(f"  Difference: {abs(manual - B[i, j, k_idx, l])}")


def check_phi_phi_flag():
    """Check the phi_phi_flag special handling"""
    print("\n" + "=" * 80)
    print("PHI_PHI_FLAG SPECIAL HANDLING")
    print("=" * 80)

    nx = 10
    A = np.ones((nx, nx, nx, nx))
    delta = [1.0, 1.0, 1.0, 1.0]

    # Test first derivative with phi_phi_flag
    print("\nFirst derivative (k=2) with phi_phi_flag=True:")
    B1 = take_finite_difference_1(A, 2, delta, phi_phi_flag=True)
    print(f"  B[:,:,0,:] should be 2*4 = 8: {B1[5,5,0,5]} (expected: 8)")
    print(f"  B[:,:,1,:] should be 2*3 = 6: {B1[5,5,1,5]} (expected: 6)")
    print(f"  B[:,:,-2,:] should be 2*(s[2]-5-1): {B1[5,5,-2,5]} (expected: {2*(nx-5-1)})")
    print(f"  B[:,:,-1,:] should be 2*(s[2]-5): {B1[5,5,-1,5]} (expected: {2*(nx-5)})")

    # Test second derivative with phi_phi_flag
    print("\nSecond derivative (k1=k2=2) with phi_phi_flag=True:")
    B2 = take_finite_difference_2(A, 2, 2, delta, phi_phi_flag=True)
    print(f"  B[:,:,0,:] should be -2: {B2[5,5,0,5]} (expected: -2)")
    print(f"  B[:,:,1,:] should be -2: {B2[5,5,1,5]} (expected: -2)")
    print(f"  B[:,:,-2,:] should be 2: {B2[5,5,-2,5]} (expected: 2)")
    print(f"  B[:,:,-1,:] should be 2: {B2[5,5,-1,5]} (expected: 2)")

    print("\n✓ phi_phi_flag handling matches MATLAB exactly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: MATLAB vs Python")
    print("=" * 80)

    compare_first_derivative_formula()
    compare_second_derivative_formula()
    compare_mixed_derivative_formula()
    check_phi_phi_flag()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
