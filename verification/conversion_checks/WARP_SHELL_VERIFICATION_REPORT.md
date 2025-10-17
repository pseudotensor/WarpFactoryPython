# Warp Shell Metric Verification Report

**Date:** 2025-10-16
**MATLAB Source:** `/WarpFactory/Metrics/WarpShell/metricGet_WarpShellComoving.m`
**Python Source:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/warp_shell.py`

---

## Executive Summary

A comprehensive line-by-line comparison of the MATLAB and Python implementations of the warp shell comoving metric has been completed. The verification covered:

1. TOV equation implementation
2. Smoothing algorithms
3. Coordinate transformations (spherical to Cartesian)
4. Shift vector implementation
5. Physical constants (c, G, π)
6. Metric component calculations
7. Sign conventions and metric signatures

**Result:** **2 CRITICAL BUGS FOUND** + 2 minor inconsistencies

---

## Critical Bugs Found

### Bug #1: Legendre Interpolation Indexing Error (HIGH SEVERITY)

**Location:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/utils.py`, lines 133-136

**Issue:** The Python implementation incorrectly translates MATLAB's 1-based indexing to Python's 0-based indexing.

**MATLAB Code (1-based):**
```matlab
x0 = floor(r/rScale-1);
x1 = floor(r/rScale);
x2 = ceil(r/rScale);
x3 = ceil(r/rScale+1);

y0 = inputArray(max(x0,1));  % Access 1st, 2nd, 3rd, 4th elements
y1 = inputArray(max(x1,1));
y2 = inputArray(max(x2,1));
y3 = inputArray(max(x3,1));
```

**Current Python Code (INCORRECT):**
```python
x0 = int(np.floor(r/r_scale - 1))
x1 = int(np.floor(r/r_scale))
x2 = int(np.ceil(r/r_scale))
x3 = int(np.ceil(r/r_scale + 1))

y0 = input_array[max(x0, 0)]  # WRONG! Off by one!
y1 = input_array[max(x1, 0)]
y2 = input_array[max(x2, 0)]
y3 = input_array[max(x3, 0)]
```

**Example showing the bug:**
- At r = 2.5:
  - MATLAB calculates: x0=1, x1=2, x2=3, x3=4
  - MATLAB accesses: inputArray(1), inputArray(2), inputArray(3), inputArray(4) = [1.0, 2.0, 3.0, 4.0]
  - Python calculates: x0=1, x1=2, x2=3, x3=4 (same values, but these are MATLAB indices!)
  - Python INCORRECTLY accesses: input_array[1], input_array[2], input_array[3], input_array[4] = [2.0, 3.0, 4.0, 5.0]
  - Result: All interpolated values are shifted by +1!

**Impact:**
- **Affects ALL metric tensor components across the entire grid**
- Every interpolated value (A, B, shift vector) is systematically wrong
- Metric will not match MATLAB output
- Physical predictions will be quantitatively incorrect

**Fix:**
```python
# Convert MATLAB 1-based indices to Python 0-based
y0 = input_array[max(x0 - 1, 0)]
y1 = input_array[max(x1 - 1, 0)]
y2 = input_array[max(x2 - 1, 0)]
y3 = input_array[max(x3 - 1, 0)]
```

---

### Bug #2: Smoothing Algorithm Implementation Mismatch (MEDIUM SEVERITY)

**Location:** `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/utils.py`, lines 207-240

**Issue:** Python uses Savitzky-Golay filter while MATLAB uses moving average, producing different smoothed profiles.

**MATLAB Code:**
```matlab
rho = smooth(smooth(smooth(smooth(rho,1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor),1.79*smoothFactor);
P = smooth(smooth(smooth(smooth(P,smoothFactor),smoothFactor),smoothFactor),smoothFactor);
```

MATLAB's `smooth()` function with a single parameter performs **moving average** smoothing.

**Current Python Code:**
```python
def smooth_array(arr: np.ndarray, smooth_factor: float, iterations: int = 4) -> np.ndarray:
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
            result = savgol_filter(result, window_length, polyorder)  # WRONG METHOD!

    return result
```

**Problems:**
1. Uses Savitzky-Golay filter instead of moving average
2. Has double application of 1.79 factor (in function call AND inside function)
3. Different window length calculation

**Impact:**
- Density profile (rho) will be smoothed differently
- Pressure profile (P) will be smoothed differently
- Mass distribution M (integrated from rho) will differ
- Metric components A and B (dependent on M and P) will differ quantitatively

**Fix:**
```python
def smooth_array(arr: np.ndarray, smooth_factor: float, iterations: int = 1) -> np.ndarray:
    """
    Smooth an array using moving average to match MATLAB's smooth() function.

    Args:
        arr: Input array
        smooth_factor: Window size for moving average
        iterations: Number of passes (default: 1)

    Returns:
        Smoothed array
    """
    from scipy.ndimage import uniform_filter1d

    result = arr.copy()
    window_size = int(smooth_factor)
    if window_size < 1:
        window_size = 1

    for _ in range(iterations):
        result = uniform_filter1d(result, size=window_size, mode='nearest')

    return result
```

Then update calls:
```python
# For density (4 iterations with 1.79*smooth_factor window)
rho_smooth = smooth_array(rho, 1.79 * smooth_factor, iterations=4)

# For pressure (4 iterations with smooth_factor window)
P_smooth = smooth_array(P, smooth_factor, iterations=4)
```

---

## Minor Issues

### Issue #3: Epsilon Value Inconsistency (LOW SEVERITY)

**Location:**
- MATLAB: line 131
- Python: line 134

**MATLAB:**
```matlab
epsilon = 0;
```

**Python:**
```python
epsilon = 1e-10
```

**Impact:** Negligible numerical difference (1e-10 is essentially zero), but inconsistent with MATLAB.

**Fix:** Change Python to `epsilon = 0`

---

### Issue #4: Smooth Window Length Double Factoring (MEDIUM SEVERITY)

This is part of Bug #2 above. The 1.79 factor is applied twice:
1. When calling the function: `smooth_array(rho, 1.79 * smooth_factor, iterations=4)`
2. Inside the function: `window_length = max(5, int(1.79 * smooth_factor))`

This is fixed by the proposed solution for Bug #2.

---

## Verified Correct Components

The following components were verified to be correctly implemented:

### ✓ Physical Constants
- **c (speed of light):** 2.99792458e8 m/s - EXACT MATCH
- **G (gravitational constant):** 6.67430e-11 m³/kg/s² - EXACT MATCH

### ✓ TOV Equation Implementation
```python
# Python (correct)
numerator = R * np.sqrt(R - 2*G*M_end/c**2) - np.sqrt(R**3 - 2*G*M_end*r**2/c**2)
denominator = np.sqrt(R**3 - 2*G*M_end*r**2/c**2) - 3*R*np.sqrt(R - 2*G*M_end/c**2)
P = c**2 * rho * (numerator / denominator) * (r < R)
```

Matches MATLAB formula exactly.

### ✓ Compact Sigmoid Function
```python
# Python (correct)
exponent = ((R2 - R1 - 2*Rbuff) * (sigma + 2) / 2 *
            (1.0 / (r - R2 + Rbuff) + 1.0 / (r - R1 - Rbuff)))
f = np.abs(1.0 / (np.exp(exponent) + 1) *
           (r > R1 + Rbuff) * (r < R2 - Rbuff) +
           (r >= R2 - Rbuff) - 1)
```

Matches MATLAB formula exactly.

### ✓ Alpha Numeric Solver
```python
# Python (correct)
dalpha = (G*M/c**2 + 4*np.pi*G*r**3*P/c**4) / (r**2 - 2*G*M*r/c**2)
dalpha[0] = 0
alpha_temp = cumulative_trapezoid(dalpha, r, initial=0)
C = 0.5 * np.log(1 - 2*G*M[-1]/r[-1]/c**2)
offset = C - alpha_temp[-1]
alpha = alpha_temp + offset
```

Matches MATLAB implementation exactly, including boundary condition.

### ✓ Metric Component B
```python
# Python (correct)
B = 1.0 / (1.0 - 2*G*M / (rsample * c**2))
B[0] = 1.0
```

Equivalent to MATLAB: `B = (1-2*G.*M./rsample/c^2).^(-1); B(1) = 1;`

### ✓ Metric Component A
```python
# Python (correct)
A = -np.exp(2.0 * a)
```

Matches MATLAB: `A = -exp(2.*a);`
Negative sign correctly implements (-+++) signature.

### ✓ Spherical to Cartesian Transformation
All transformation formulas verified correct:
```python
g11_cart = g11_sph
g22_cart = E * cos_phi**2 * sin_theta**2 + cos_phi**2 * cos_theta**2 + sin_phi**2
g33_cart = E * sin_phi**2 * sin_theta**2 + cos_theta**2 * sin_phi**2 + cos_phi**2
g44_cart = E * cos_theta**2 + sin_theta**2
g23_cart = E * cos_phi * sin_phi * sin_theta**2 + cos_phi * cos_theta**2 * sin_phi - cos_phi * sin_phi
g24_cart = E * cos_phi * cos_theta * sin_theta - cos_phi * cos_theta * sin_theta
g34_cart = E * cos_theta * sin_phi * sin_theta - cos_theta * sin_phi * sin_theta
```

Matches MATLAB `sph2cartDiag.m` exactly, including special angle handling.

### ✓ Coordinate Indexing Conversion
```python
# Python (correct)
for i in range(grid_size[1]):
    x = (i + 1) * grid_scaling[1] - world_center[1]
```

Correctly converts MATLAB's 1-based loop:
```matlab
for i = 1:gridSize(2)
    x = (i*gridScaling(2)-worldCenter(2))
```

### ✓ Shift Vector Application
```python
# Python (correct)
metric_dict[(0, 1)] = metric_dict[(0, 1)] - metric_dict[(0, 1)] * shift_matrix - shift_matrix * v_warp
metric_dict[(1, 0)] = metric_dict[(0, 1)]
```

Matches MATLAB exactly:
```matlab
Metric.tensor{1,2} = Metric.tensor{1,2}-Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp;
Metric.tensor{2,1} = Metric.tensor{1,2};
```

---

## Summary

**Critical Bugs:** 2
1. **Legendre interpolation indexing** (HIGH) - All metric values wrong
2. **Smoothing algorithm mismatch** (MEDIUM) - Quantitative differences in profiles

**Minor Issues:** 2
3. Epsilon value inconsistency (LOW)
4. Smooth window double factoring (part of #2)

**Verified Correct:** 9 major components
- TOV equation
- Compact sigmoid
- Alpha solver
- Metric components A and B
- Coordinate transformations
- Physical constants
- Shift vector application
- Coordinate indexing
- Sign conventions

---

## Recommendations

1. **IMMEDIATELY FIX Bug #1** - This is a fundamental indexing error affecting all outputs
2. **Fix Bug #2** - Replace Savitzky-Golay with moving average
3. **Fix epsilon value** for consistency
4. **Add regression tests** comparing Python output to MATLAB reference calculations
5. **Add unit tests** for `legendre_radial_interp` with known values

---

## Test Files Generated

- `/WarpFactory/warpfactory_py/test_warp_shell_verification.py` - Full test suite
- `/WarpFactory/warpfactory_py/test_legendre_debug.py` - Debug script for interpolation
- `/WarpFactory/warpfactory_py/test_legendre_careful.py` - Detailed indexing analysis
- `/WarpFactory/warpfactory_py/final_comparison_report.py` - Summary report

All tests confirm the identified bugs and verify correct components.
