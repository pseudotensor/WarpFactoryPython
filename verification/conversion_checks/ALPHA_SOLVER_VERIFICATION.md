# Alpha (Lapse Function) Calculation Verification

## Summary

**STATUS: ✓ VERIFIED - No bugs found**

The Python implementation of `alpha_numeric_solver` in `/WarpFactory/warpfactory_py/warpfactory/metrics/warp_shell/utils.py` correctly matches the MATLAB implementation in `/WarpFactory/Metrics/utils/alphaNumericSolver.m`.

## Comparison

### 1. Differential Equation Formula

**MATLAB** (line 4):
```matlab
dalpha = (G*M./c^2+4*pi*G*r.^3.*P./c^4)./(r.^2-2*G*M.*r./c^2);
```

**Python** (line 97):
```python
dalpha = (G*M/c**2 + 4*np.pi*G*r**3*P/c**4) / (r**2 - 2*G*M*r/c**2)
```

**Status**: ✓ IDENTICAL

The formula implements:
```
dα/dr = (GM/c² + 4πGr³P/c⁴) / (r² - 2GMr/c²)
```

This is the correct TOV-based differential equation for the lapse function α.

### 2. Integration Method

**MATLAB** (line 6):
```matlab
alphaTemp = cumtrapz(r,dalpha);
```

**Python** (line 101):
```python
alpha_temp = cumulative_trapezoid(dalpha, r, initial=0)
```

**Status**: ✓ EQUIVALENT

Both use trapezoidal integration:
- MATLAB: `cumtrapz(x, y)` - cumulative integral of y with respect to x
- Python: `cumulative_trapezoid(y, x, initial=0)` - same operation

Note the argument order difference, but both correctly integrate dalpha over r.

### 3. Handling of r=0 Singularity

**MATLAB** (line 5):
```matlab
dalpha(1) = 0;
```

**Python** (line 98):
```python
dalpha[0] = 0
```

**Status**: ✓ IDENTICAL

Both set the first element to zero to avoid division by zero at r=0. This is necessary because:
- At r=0: dalpha = (GM/c²) / (0 - 0) → ∞
- The radial grids in both implementations start at r=0
- Setting dalpha[0] = 0 is physically reasonable since α is typically set to 0 at the origin

### 4. Boundary Condition at R

**MATLAB** (line 7):
```matlab
C = 1/2*log(1-2*G*M(end)./r(end)/c^2);
```

**Python** (line 104):
```python
C = 0.5 * np.log(1 - 2*G*M[-1]/r[-1]/c**2)
```

**Status**: ✓ IDENTICAL

The boundary condition α(R) = (1/2)ln(1 - 2GM(R)/Rc²) matches the Schwarzschild solution for the lapse function outside a spherical mass distribution.

### 5. Offset Calculation

**MATLAB** (lines 8-9):
```matlab
offset = C-alphaTemp(end);
alpha = alphaTemp+offset;
```

**Python** (lines 105-106):
```python
offset = C - alpha_temp[-1]
alpha = alpha_temp + offset
```

**Status**: ✓ IDENTICAL

Both:
1. Compute the required value C at the boundary
2. Calculate offset as C minus the integrated value at the boundary
3. Add this offset to the entire integrated function

This ensures α(R) = C exactly.

## Verification Tests

### Test 1: Schwarzschild Vacuum Solution

**Test Setup**:
- Mass: M = 1×10²⁶ kg (Jupiter mass)
- Range: r = 0 to 1×10⁹ m
- Pressure: P = 0 everywhere (vacuum)
- Grid points: 100,000

**Expected Result**:
For vacuum, the solver should reproduce the exact Schwarzschild lapse function:
```
α(r) = (1/2) ln(1 - 2GM/rc²)
```

**Actual Results**:
- Maximum relative error (middle region): 1.667×10⁻⁵
- Mean relative error: 2.724×10⁻⁷
- Boundary condition error: 4.326×10⁻²²

**Status**: ✓ PASS (within tolerance for trapezoidal integration)

The small error is due to:
1. Trapezoidal integration (2nd order accurate)
2. Finite grid spacing
3. Numerical precision near endpoints

### Test 2: Warp Shell Configuration

**Test Setup**:
- Shell mass: 1×10⁶ kg distributed between R1=8m and R2=10m
- Non-zero pressure from TOV equation
- Range: 0 to 18m
- Grid points: 100,000

**Results**:
- Boundary condition: Exact match (error ~ 10⁻²²)
- All values finite: Yes
- Smoothness: Confirmed (low variance in dα/dr)

**Status**: ✓ PASS

## Indexing Verification

**MATLAB**: 1-based indexing
- `M(end)` = last element
- `M(1)` = first element

**Python**: 0-based indexing
- `M[-1]` = last element
- `M[0]` = first element

**Status**: ✓ CORRECTLY HANDLED

All array indexing is correctly translated between the two languages.

## Physical Constants

Both implementations use consistent physical constants:
- c (speed of light): 2.99792458×10⁸ m/s
- G (gravitational constant): 6.67430×10⁻¹¹ m³/kg/s²

## Conclusion

✓ **NO BUGS FOUND**

The Python implementation `alpha_numeric_solver` is a correct translation of the MATLAB `alphaNumericSolver`. All aspects are verified:

1. ✓ Differential equation dα/dr formula is correct
2. ✓ Trapezoidal integration method is correct
3. ✓ Handling of r=0 singularity is correct
4. ✓ Boundary condition at R is correct
5. ✓ Offset calculation is correct
6. ✓ Reproduces Schwarzschild exact solution within numerical precision
7. ✓ Works correctly for warp shell configurations

The maximum error of ~10⁻⁵ in the Schwarzschild test is expected and acceptable for second-order trapezoidal integration over 100,000 points.

## Test Files

- `/WarpFactory/warpfactory_py/test_alpha_proper.py` - Main verification test
- `/WarpFactory/warpfactory_py/test_alpha_debug.py` - Detailed debugging analysis
- `/WarpFactory/warpfactory_py/test_alpha_schwarzschild.py` - Initial Schwarzschild test

Run verification:
```bash
cd /WarpFactory/warpfactory_py
python3 test_alpha_proper.py
```
