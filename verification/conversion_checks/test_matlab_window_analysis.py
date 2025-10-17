"""
Detailed analysis of MATLAB smooth() function window sizing.

MATLAB Code Analysis:
Line 84: rho = smooth(smooth(smooth(smooth(rho,1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor);
Line 88: P = smooth(smooth(smooth(smooth(P,smoothFactor),smoothFactor),smoothFactor),smoothFactor);
Line 104: shiftRadialVector = smooth(smooth(shiftRadialVector,smoothFactor),smoothFactor);

Key Question: Does MATLAB smooth(data, span) use span as the window size?
Answer: YES - MATLAB's smooth(data, span) uses span as the window length for moving average.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def matlab_smooth(data, span):
    """
    MATLAB smooth() function equivalent.

    MATLAB Documentation:
    yy = smooth(y, span) uses a moving average with span window points.
    span must be odd. If span is even, it is increased by 1.
    """
    span = int(span)
    if span < 1:
        return data
    if span % 2 == 0:
        span += 1  # MATLAB increases even spans by 1

    # MATLAB uses centered moving average with nearest padding at edges
    return uniform_filter1d(data, size=span, mode='nearest')


def analyze_window_calculation():
    """
    Analyze the window length calculation in Python vs MATLAB.
    """
    print("=" * 70)
    print("WINDOW LENGTH CALCULATION ANALYSIS")
    print("=" * 70)

    smooth_factors = [1, 2, 5, 10, 15, 20]

    print("\nFor DENSITY (1.79 * smoothFactor):")
    print(f"{'smoothFactor':<15} {'MATLAB Window':<15} {'Python Window':<15} {'Difference':<15}")
    print("-" * 60)

    for sf in smooth_factors:
        matlab_window = 1.79 * sf
        python_window = max(5, int(1.79 * sf))
        if python_window % 2 == 0:
            python_window += 1

        diff = abs(matlab_window - python_window)
        print(f"{sf:<15} {matlab_window:<15.2f} {python_window:<15} {diff:<15.2f}")

    print("\nFor PRESSURE/SHIFT (smoothFactor):")
    print(f"{'smoothFactor':<15} {'MATLAB Window':<15} {'Python Window':<15} {'Difference':<15}")
    print("-" * 60)

    for sf in smooth_factors:
        matlab_window = sf
        python_window = max(5, int(1.79 * sf))  # Python ALWAYS applies 1.79 multiplier!
        if python_window % 2 == 0:
            python_window += 1

        diff = abs(matlab_window - python_window)
        print(f"{sf:<15} {matlab_window:<15.2f} {python_window:<15} {diff:<15.2f}")


def test_actual_smoothing_behavior():
    """
    Test the actual smoothing behavior with a step function.
    """
    print("\n" + "=" * 70)
    print("ACTUAL SMOOTHING BEHAVIOR TEST")
    print("=" * 70)

    # Create a simple step function
    n = 1000
    data = np.zeros(n)
    data[400:600] = 1.0

    smooth_factor = 10

    # MATLAB approach for density: window = 1.79 * smoothFactor
    matlab_window_density = 1.79 * smooth_factor
    matlab_smoothed_density = data.copy()
    for _ in range(4):
        matlab_smoothed_density = matlab_smooth(matlab_smoothed_density, matlab_window_density)

    # Python approach: window = int(1.79 * smooth_factor), Savitzky-Golay
    python_window = max(5, int(1.79 * smooth_factor))
    if python_window % 2 == 0:
        python_window += 1
    polyorder = min(3, python_window - 1)

    python_smoothed = data.copy()
    for _ in range(4):
        python_smoothed = savgol_filter(python_smoothed, python_window, polyorder)

    # Compute transition widths
    # Find where signal goes from 10% to 90%
    threshold_low = 0.1
    threshold_high = 0.9

    # MATLAB transition
    idx_matlab_low = np.where(matlab_smoothed_density > threshold_low)[0][0]
    idx_matlab_high = np.where(matlab_smoothed_density > threshold_high)[0][0]
    transition_width_matlab = idx_matlab_high - idx_matlab_low

    # Python transition
    idx_python_low = np.where(python_smoothed > threshold_low)[0][0]
    idx_python_high = np.where(python_smoothed > threshold_high)[0][0]
    transition_width_python = idx_python_high - idx_python_low

    print(f"\nSmooth Factor: {smooth_factor}")
    print(f"MATLAB window: {matlab_window_density:.1f}")
    print(f"Python window: {python_window}")
    print(f"\nTransition width (10% to 90%):")
    print(f"  MATLAB: {transition_width_matlab} points")
    print(f"  Python: {transition_width_python} points")
    print(f"  Difference: {abs(transition_width_matlab - transition_width_python)} points")

    # Peak values
    print(f"\nPeak values after smoothing:")
    print(f"  Original: {data.max():.6f}")
    print(f"  MATLAB: {matlab_smoothed_density.max():.6f}")
    print(f"  Python: {python_smoothed.max():.6f}")
    print(f"  Difference: {abs(matlab_smoothed_density.max() - python_smoothed.max()):.6f}")

    return data, matlab_smoothed_density, python_smoothed


def identify_critical_issue():
    """
    Identify the critical issue in the Python implementation.
    """
    print("\n" + "=" * 70)
    print("CRITICAL ISSUE IDENTIFICATION")
    print("=" * 70)

    print("\nPROBLEM FOUND IN utils.py line 225:")
    print("  window_length = max(5, int(1.79 * smooth_factor))")
    print("\nThis means Python ALWAYS applies the 1.79 multiplier!")
    print("\nBut MATLAB code shows:")
    print("  Line 84: smooth(rho, 1.79*smoothFactor)  <- density gets 1.79x")
    print("  Line 88: smooth(P, smoothFactor)         <- pressure gets 1.0x")
    print("  Line 104: smooth(shift, smoothFactor)    <- shift gets 1.0x")
    print("\nPython should accept smooth_factor directly, not always multiply by 1.79!")

    print("\nLooking at warp_shell.py line 102-103:")
    print("  Line 102: rho_smooth = smooth_array(rho, 1.79 * smooth_factor, iterations=4)")
    print("  Line 103: P_smooth = smooth_array(P, smooth_factor, iterations=4)")

    print("\nSo Python compensates by passing 1.79*smooth_factor for density.")
    print("But then smooth_array() MULTIPLIES BY 1.79 AGAIN!")
    print("This gives: 1.79 * 1.79 = 3.2041x window size for density!")
    print("\nThis is INCORRECT and explains discrepancies!")


def test_double_multiplication_bug():
    """
    Demonstrate the double multiplication bug.
    """
    print("\n" + "=" * 70)
    print("DOUBLE MULTIPLICATION BUG DEMONSTRATION")
    print("=" * 70)

    smooth_factor = 10

    print(f"\nFor smooth_factor = {smooth_factor}:")
    print("\nMALTAB Implementation:")
    print(f"  Density window: 1.79 * {smooth_factor} = {1.79 * smooth_factor:.1f}")
    print(f"  Pressure window: 1.0 * {smooth_factor} = {smooth_factor:.1f}")

    print("\nPython INTENDED:")
    print(f"  Density: smooth_array(rho, 1.79 * {smooth_factor}) = {1.79 * smooth_factor:.1f}")
    print(f"  Pressure: smooth_array(P, {smooth_factor}) = {smooth_factor:.1f}")

    print("\nPython ACTUAL (with bug in line 225):")
    python_density_factor = 1.79 * smooth_factor
    python_density_window = int(1.79 * python_density_factor)
    python_pressure_window = int(1.79 * smooth_factor)

    print(f"  Density: int(1.79 * {python_density_factor:.1f}) = {python_density_window}")
    print(f"  Pressure: int(1.79 * {smooth_factor}) = {python_pressure_window}")

    print("\nEFFECT:")
    matlab_density = 1.79 * smooth_factor
    python_density = python_density_window

    print(f"  Density window: {matlab_density:.1f} (MATLAB) vs {python_density} (Python)")
    print(f"  RATIO: {python_density / matlab_density:.2f}x")
    print(f"  This is {(python_density / matlab_density - 1) * 100:.1f}% LARGER!")


def propose_fix():
    """
    Propose the fix for the smoothing function.
    """
    print("\n" + "=" * 70)
    print("PROPOSED FIX")
    print("=" * 70)

    print("\nIn utils.py, smooth_array() function:")
    print("\nCURRENT CODE (lines 223-227):")
    print("""
    # Determine window length based on smooth factor
    # MATLAB's smooth() uses a moving average; we use Savitzky-Golay for similar effect
    window_length = max(5, int(1.79 * smooth_factor))
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    """)

    print("\nPROPOSED FIX:")
    print("""
    # Determine window length based on smooth factor
    # MATLAB's smooth() uses a moving average; we use Savitzky-Golay for similar effect
    # NOTE: smooth_factor is passed directly as the window span (no multiplier here)
    window_length = max(5, int(smooth_factor))
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    """)

    print("\nThis removes the 1.79 multiplier from smooth_array().")
    print("The multiplier should only be applied when CALLING smooth_array(),")
    print("which is already done correctly in warp_shell.py line 102.")

    print("\n" + "=" * 70)
    print("ADDITIONAL CONCERN: Moving Average vs Savitzky-Golay")
    print("=" * 70)

    print("\nMALTAB uses: Moving Average (uniform weighting)")
    print("Python uses: Savitzky-Golay (polynomial fit)")
    print("\nThese are fundamentally DIFFERENT filters:")
    print("  - Moving avg smooths equally across window")
    print("  - Savitzky-Golay fits polynomial, preserves features better")
    print("\nFor critical warp shell calculations, this difference matters!")
    print("\nRECOMMENDATION: Use scipy.ndimage.uniform_filter1d() instead")
    print("to exactly match MATLAB's moving average behavior.")


if __name__ == "__main__":
    analyze_window_calculation()
    test_actual_smoothing_behavior()
    identify_critical_issue()
    test_double_multiplication_bug()
    propose_fix()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
