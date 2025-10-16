"""
Test script to compare MATLAB smooth() vs Python Savitzky-Golay smoothing.

This script verifies whether the Python implementation's Savitzky-Golay filter
provides equivalent smoothing to MATLAB's moving average smooth() function.

MATLAB implementation:
- Uses smooth(data, window_span) which is a moving average filter
- For density: 4 iterations with window = 1.79 * smoothFactor
- For pressure: 4 iterations with window = smoothFactor
- For shift vector: 2 iterations with window = smoothFactor

Python implementation:
- Uses savgol_filter with window_length = int(1.79 * smooth_factor), polyorder=3
- Same iteration counts as MATLAB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def matlab_smooth_equivalent(data, span):
    """
    Equivalent to MATLAB's smooth(data, span) using moving average.

    MATLAB's smooth() with a single argument uses a moving average filter.
    """
    # MATLAB's smooth uses centered moving average
    # Handle edge cases by using 'nearest' mode padding
    if span < 2:
        return data
    return uniform_filter1d(data, size=int(span), mode='nearest')


def python_smooth_array(arr, smooth_factor, iterations=4):
    """
    Current Python implementation using Savitzky-Golay filter.
    """
    result = arr.copy()

    window_length = max(5, int(1.79 * smooth_factor))
    if window_length % 2 == 0:
        window_length += 1

    window_length = min(window_length, len(result))
    if window_length < 5:
        window_length = 5

    polyorder = min(3, window_length - 1)

    for _ in range(iterations):
        if len(result) > window_length:
            result = savgol_filter(result, window_length, polyorder)

    return result


def matlab_smooth_array(arr, smooth_factor, iterations=4):
    """
    MATLAB-equivalent implementation using moving average.
    """
    result = arr.copy()
    span = 1.79 * smooth_factor  # Note: MATLAB code uses 1.79*smoothFactor

    for _ in range(iterations):
        result = matlab_smooth_equivalent(result, span)

    return result


def create_test_signals():
    """
    Create synthetic test signals similar to warp shell profiles.
    """
    n_points = 100000
    r = np.linspace(0, 1000, n_points)

    # Test 1: Step function (density-like profile)
    R1, R2 = 300, 500
    rho = np.zeros(n_points)
    rho[(r > R1) & (r < R2)] = 1.0

    # Test 2: Smooth sigmoid (shift vector-like)
    Rbuff = 20
    sigma = 0.5
    exponent = ((R2 - R1 - 2*Rbuff) * (sigma + 2) / 2 *
                (1.0 / (r - R2 + Rbuff + 0.01) + 1.0 / (r - R1 - Rbuff + 0.01)))
    shift = np.abs(1.0 / (np.exp(exponent) + 1) *
                   (r > R1 + Rbuff) * (r < R2 - Rbuff) +
                   (r >= R2 - Rbuff) - 1)
    shift[np.isinf(shift)] = 0
    shift[np.isnan(shift)] = 0

    # Test 3: Noisy sine wave
    noisy_sine = np.sin(2 * np.pi * r / 200) + 0.1 * np.random.randn(n_points)

    return r, rho, shift, noisy_sine


def compute_differences(original, matlab_smooth, python_smooth, name):
    """
    Compute and report differences between smoothing methods.
    """
    # Compute various error metrics
    mae = np.mean(np.abs(matlab_smooth - python_smooth))
    rmse = np.sqrt(np.mean((matlab_smooth - python_smooth)**2))
    max_error = np.max(np.abs(matlab_smooth - python_smooth))

    # Relative error (avoid division by zero)
    nonzero_mask = np.abs(matlab_smooth) > 1e-10
    if np.any(nonzero_mask):
        rel_error = np.mean(np.abs((matlab_smooth[nonzero_mask] -
                                    python_smooth[nonzero_mask]) /
                                   matlab_smooth[nonzero_mask])) * 100
    else:
        rel_error = 0.0

    # Compute gradient differences (important for pressure/density)
    grad_matlab = np.gradient(matlab_smooth)
    grad_python = np.gradient(python_smooth)
    grad_mae = np.mean(np.abs(grad_matlab - grad_python))

    print(f"\n{name} Smoothing Comparison:")
    print(f"  Mean Absolute Error: {mae:.6e}")
    print(f"  Root Mean Square Error: {rmse:.6e}")
    print(f"  Maximum Error: {max_error:.6e}")
    print(f"  Mean Relative Error: {rel_error:.4f}%")
    print(f"  Gradient MAE: {grad_mae:.6e}")

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'rel_error': rel_error,
        'grad_mae': grad_mae
    }


def test_smoothing_comparison():
    """
    Main test function comparing MATLAB and Python smoothing.
    """
    print("=" * 70)
    print("WARP SHELL SMOOTHING COMPARISON: MATLAB vs Python")
    print("=" * 70)

    # Generate test signals
    r, rho, shift, noisy_sine = create_test_signals()

    # Test different smooth factors (typical values used in warp shell)
    smooth_factors = [1, 5, 10, 20]

    results = {}

    for smooth_factor in smooth_factors:
        print(f"\n{'='*70}")
        print(f"Testing with smooth_factor = {smooth_factor}")
        print(f"{'='*70}")

        # Test 1: Density profile (4 iterations, window = 1.79 * smooth_factor)
        rho_matlab = matlab_smooth_array(rho, smooth_factor, iterations=4)
        rho_python = python_smooth_array(rho, smooth_factor, iterations=4)
        results[f'rho_{smooth_factor}'] = compute_differences(
            rho, rho_matlab, rho_python, f"Density (SF={smooth_factor})")

        # Test 2: Shift vector (2 iterations, window = smooth_factor)
        # Adjust for different iteration count
        shift_matlab = matlab_smooth_array(shift, smooth_factor/1.79, iterations=2)
        # For Python, we need to account for the 1.79 multiplier in the function
        shift_python_temp = shift.copy()
        window_length = max(5, int(smooth_factor))
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(shift))
        polyorder = min(3, window_length - 1)
        for _ in range(2):
            if len(shift_python_temp) > window_length:
                shift_python_temp = savgol_filter(shift_python_temp, window_length, polyorder)

        results[f'shift_{smooth_factor}'] = compute_differences(
            shift, shift_matlab, shift_python_temp,
            f"Shift Vector (SF={smooth_factor})")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\nKey Differences:")
    print("1. MATLAB smooth() = Moving Average Filter (rectangular window)")
    print("2. Python savgol_filter = Savitzky-Golay Filter (polynomial fit)")
    print("\nImplications:")
    print("- Moving average: Equal weighting, can create steps at edges")
    print("- Savitzky-Golay: Preserves peaks/valleys better, polynomial fit")
    print("- SG filter better preserves shape features but may differ in magnitude")

    # Check for critical differences
    critical_errors = []
    for key, result in results.items():
        if result['rel_error'] > 5.0:  # More than 5% relative error
            critical_errors.append((key, result['rel_error']))

    if critical_errors:
        print("\nWARNING: Significant differences detected:")
        for key, error in critical_errors:
            print(f"  {key}: {error:.2f}% relative error")
    else:
        print("\nNo critical differences detected (all < 5% relative error)")

    return results


def visualize_comparison(smooth_factor=10):
    """
    Create visualizations comparing the two smoothing methods.
    """
    print(f"\nGenerating comparison plots for smooth_factor={smooth_factor}...")

    r, rho, shift, noisy_sine = create_test_signals()

    # Smooth using both methods
    rho_matlab = matlab_smooth_array(rho, smooth_factor, iterations=4)
    rho_python = python_smooth_array(rho, smooth_factor, iterations=4)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Full profiles
    ax = axes[0]
    ax.plot(r, rho, 'k-', alpha=0.3, linewidth=0.5, label='Original')
    ax.plot(r, rho_matlab, 'b-', linewidth=2, label='MATLAB (Moving Avg)')
    ax.plot(r, rho_python, 'r--', linewidth=2, label='Python (Savitzky-Golay)')
    ax.set_xlabel('Radius r')
    ax.set_ylabel('Density ρ')
    ax.set_title(f'Density Profile Smoothing Comparison (smooth_factor={smooth_factor})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoomed view of edge
    ax = axes[1]
    zoom_start, zoom_end = 29500, 30500
    ax.plot(r[zoom_start:zoom_end], rho[zoom_start:zoom_end],
            'k-', alpha=0.3, linewidth=1, label='Original')
    ax.plot(r[zoom_start:zoom_end], rho_matlab[zoom_start:zoom_end],
            'b-', linewidth=2, label='MATLAB (Moving Avg)')
    ax.plot(r[zoom_start:zoom_end], rho_python[zoom_start:zoom_end],
            'r--', linewidth=2, label='Python (Savitzky-Golay)')
    ax.set_xlabel('Radius r')
    ax.set_ylabel('Density ρ')
    ax.set_title('Zoomed View: Inner Edge Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Difference plot
    ax = axes[2]
    difference = rho_python - rho_matlab
    ax.plot(r, difference, 'purple', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Radius r')
    ax.set_ylabel('Difference (Python - MATLAB)')
    ax.set_title('Smoothing Method Difference')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/WarpFactory/warpfactory_py/smoothing_comparison.png', dpi=150)
    print(f"Plot saved to: /WarpFactory/warpfactory_py/smoothing_comparison.png")

    return fig


if __name__ == "__main__":
    # Run comparison tests
    results = test_smoothing_comparison()

    # Generate visualizations
    try:
        visualize_comparison(smooth_factor=10)
    except Exception as e:
        print(f"\nNote: Visualization skipped ({e})")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
