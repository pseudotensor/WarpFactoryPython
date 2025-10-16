# Required Fixes for Warp Shell Metric Implementation

## Fix #1: Legendre Interpolation Indexing (HIGH PRIORITY)

**File:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/utils.py`

**Lines:** 133-136

**Current Code (INCORRECT):**
```python
# Get values (with bounds checking)
y0 = input_array[max(x0, 0)]
y1 = input_array[max(x1, 0)]
y2 = input_array[max(x2, 0)]
y3 = input_array[max(x3, 0)]
```

**Fixed Code:**
```python
# Get values (with bounds checking)
# Note: x0-x3 are calculated as MATLAB indices (1-based)
# Must convert to Python indices (0-based) by subtracting 1
y0 = input_array[max(x0 - 1, 0)]
y1 = input_array[max(x1 - 1, 0)]
y2 = input_array[max(x2 - 1, 0)]
y3 = input_array[max(x3 - 1, 0)]
```

**Explanation:**
The index calculations (`floor(r - 1)`, etc.) produce MATLAB-style 1-based indices. These must be converted to Python's 0-based indexing by subtracting 1 before accessing the array.

---

## Fix #2: Smoothing Algorithm (MEDIUM PRIORITY)

**File:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/utils.py`

**Lines:** 207-240

**Current Code (INCORRECT):**
```python
def smooth_array(arr: np.ndarray, smooth_factor: float, iterations: int = 4) -> np.ndarray:
    """
    Smooth an array using Savitzky-Golay filter

    Applies multiple passes of smoothing to approximate MATLAB's smooth() function.

    Args:
        arr: Input array to smooth
        smooth_factor: Smoothing strength factor
        iterations: Number of smoothing passes (default: 4)

    Returns:
        Smoothed array
    """
    result = arr.copy()

    # Determine window length based on smooth factor
    # MATLAB's smooth() uses a moving average; we use Savitzky-Golay for similar effect
    window_length = max(5, int(1.79 * smooth_factor))
    if window_length % 2 == 0:
        window_length += 1  # Must be odd

    # Ensure window isn't larger than array
    window_length = min(window_length, len(result))
    if window_length < 5:
        window_length = 5

    polyorder = min(3, window_length - 1)

    for _ in range(iterations):
        if len(result) > window_length:
            result = savgol_filter(result, window_length, polyorder)

    return result
```

**Fixed Code:**
```python
def smooth_array(arr: np.ndarray, smooth_factor: float, iterations: int = 1) -> np.ndarray:
    """
    Smooth an array using moving average to match MATLAB's smooth() function

    MATLAB's smooth() function performs moving average smoothing.
    This implementation uses scipy's uniform_filter1d to replicate that behavior.

    Args:
        arr: Input array to smooth
        smooth_factor: Window size for moving average
        iterations: Number of smoothing passes (default: 1)

    Returns:
        Smoothed array
    """
    from scipy.ndimage import uniform_filter1d

    result = arr.copy()

    # Convert smooth_factor to integer window size
    window_size = int(smooth_factor)
    if window_size < 1:
        window_size = 1

    # Ensure window size doesn't exceed array length
    window_size = min(window_size, len(result))

    # Apply moving average filter for specified iterations
    for _ in range(iterations):
        result = uniform_filter1d(result, size=window_size, mode='nearest')

    return result
```

**Also update the function calls in warp_shell.py (lines 102-103):**

**Current:**
```python
rho_smooth = smooth_array(rho, 1.79 * smooth_factor, iterations=4)
P_smooth = smooth_array(P, smooth_factor, iterations=4)
```

**Fixed:**
```python
# Apply smoothing 4 times as in MATLAB
# For density: smooth(smooth(smooth(smooth(rho, 1.79*smoothFactor), ...), ...), ...)
rho_smooth = smooth_array(rho, 1.79 * smooth_factor, iterations=4)

# For pressure: smooth(smooth(smooth(smooth(P, smoothFactor), ...), ...), ...)
P_smooth = smooth_array(P, smooth_factor, iterations=4)
```

**Update shift vector smoothing call in warp_shell.py (line 111):**

**Current:**
```python
shift_radial_vector = smooth_array(shift_radial_vector, smooth_factor, iterations=2)
```

**Fixed:**
```python
# Apply smoothing 2 times as in MATLAB: smooth(smooth(shiftRadialVector, smoothFactor), smoothFactor)
shift_radial_vector = smooth_array(shift_radial_vector, smooth_factor, iterations=2)
```

---

## Fix #3: Epsilon Value (LOW PRIORITY)

**File:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/warp_shell.py`

**Line:** 134

**Current Code:**
```python
epsilon = 1e-10
```

**Fixed Code:**
```python
epsilon = 0
```

**Explanation:**
Match MATLAB's epsilon value for consistency. The difference is negligible numerically, but maintaining exact consistency aids in validation and comparison.

---

## Testing the Fixes

After applying these fixes, run the verification test:

```bash
cd /WarpFactory/warpfactory_py
python test_warp_shell_verification.py
```

Expected results:
- All constants should match
- TOV equation results should match
- Legendre interpolation should now return correct values at integer and fractional indices
- Smoothing should produce similar profiles to MATLAB

For more detailed validation, compare outputs between MATLAB and Python implementations on identical test cases.

---

## Import Updates Required

Add to the imports section of `utils.py`:
```python
from scipy.ndimage import uniform_filter1d
```

And remove if no longer needed:
```python
from scipy.signal import savgol_filter  # Can be removed if not used elsewhere
```

---

## Summary of Changes

1. **utils.py, legendre_radial_interp()**: Add `- 1` to array index conversions (4 lines)
2. **utils.py, smooth_array()**: Replace Savitzky-Golay with moving average (complete rewrite)
3. **warp_shell.py**: Change epsilon from 1e-10 to 0 (1 line)
4. **utils.py**: Update imports (1 addition, 1 potential removal)

Total lines changed: ~50 lines
Complexity: Low (straightforward fixes)
Risk: Low (well-defined changes with clear test cases)
