"""Debug Legendre interpolation"""
import numpy as np

def legendre_radial_interp_matlab_style(inputArray, r):
    """MATLAB-style implementation (1-based indexing)"""
    rScale = 1

    # MATLAB indices (1-based)
    x0_matlab = int(np.floor(r/rScale - 1))
    x1_matlab = int(np.floor(r/rScale))
    x2_matlab = int(np.ceil(r/rScale))
    x3_matlab = int(np.ceil(r/rScale + 1))

    print(f"MATLAB indices at r={r}: x0={x0_matlab}, x1={x1_matlab}, x2={x2_matlab}, x3={x3_matlab}")

    # MATLAB array access (convert to 0-based for Python)
    y0 = inputArray[max(x0_matlab-1, 0)]  # MATLAB max(x0,1) -> index 1 = Python index 0
    y1 = inputArray[max(x1_matlab-1, 0)]
    y2 = inputArray[max(x2_matlab-1, 0)]
    y3 = inputArray[max(x3_matlab-1, 0)]

    print(f"Values: y0={y0}, y1={y1}, y2={y2}, y3={y3}")

    x = r

    x0 = x0_matlab * rScale
    x1 = x1_matlab * rScale
    x2 = x2_matlab * rScale
    x3 = x3_matlab * rScale

    print(f"Position values: x0={x0}, x1={x1}, x2={x2}, x3={x3}")

    outputValue = (y0 * (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) +
                   y1 * (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) +
                   y2 * (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) +
                   y3 * (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2)))

    return outputValue

def legendre_radial_interp_python(inputArray, r):
    """Current Python implementation"""
    rScale = 1

    x0 = int(np.floor(r/rScale - 1))
    x1 = int(np.floor(r/rScale))
    x2 = int(np.ceil(r/rScale))
    x3 = int(np.ceil(r/rScale + 1))

    print(f"Python indices at r={r}: x0={x0}, x1={x1}, x2={x2}, x3={x3}")

    y0 = inputArray[max(x0, 0)]
    y1 = inputArray[max(x1, 0)]
    y2 = inputArray[max(x2, 0)]
    y3 = inputArray[max(x3, 0)]

    print(f"Values: y0={y0}, y1={y1}, y2={y2}, y3={y3}")

    x = r

    x0 = x0 * rScale
    x1 = x1 * rScale
    x2 = x2 * rScale
    x3 = x3 * rScale

    print(f"Position values: x0={x0}, x1={x1}, x2={x2}, x3={x3}")

    outputValue = (y0 * (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) +
                   y1 * (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) +
                   y2 * (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) +
                   y3 * (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2)))

    return outputValue

# Test array
test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

print("=" * 60)
print("Testing at r=3.0")
print("=" * 60)
print("\nMATLAB-style:")
result_matlab = legendre_radial_interp_matlab_style(test_array, 3.0)
print(f"Result: {result_matlab}")

print("\nPython current:")
result_python = legendre_radial_interp_python(test_array, 3.0)
print(f"Result: {result_python}")

print("\n" + "=" * 60)
print("Testing at r=2.5")
print("=" * 60)
print("\nMATLAB-style:")
result_matlab = legendre_radial_interp_matlab_style(test_array, 2.5)
print(f"Result: {result_matlab}")

print("\nPython current:")
result_python = legendre_radial_interp_python(test_array, 2.5)
print(f"Result: {result_python}")
