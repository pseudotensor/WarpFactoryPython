"""
Careful analysis of Legendre interpolation indexing
"""

import numpy as np

print("=" * 80)
print("CAREFUL ANALYSIS OF LEGENDRE INTERPOLATION")
print("=" * 80)
print()

# In MATLAB, legendreRadialInterp is called with a fractional index
# Let's trace through what happens in MATLAB vs Python

print("MATLAB CODE ANALYSIS:")
print("-" * 80)
print("function outputValue = legendreRadialInterp(inputArray,r)")
print("    x0 = floor(r/rScale-1);")
print("    x1 = floor(r/rScale);")
print("    x2 = ceil(r/rScale);")
print("    x3 = ceil(r/rScale+1);")
print()
print("    y0 = inputArray(max(x0,1));")
print("    y1 = inputArray(max(x1,1));")
print("    y2 = inputArray(max(x2,1));")
print("    y3 = inputArray(max(x3,1));")
print()

# Example: r = 2.5, rScale = 1
r = 2.5
rScale = 1

print(f"Example: r = {r}, rScale = {rScale}")
print()

# MATLAB calculation
x0_m = int(np.floor(r/rScale - 1))  # floor(2.5 - 1) = floor(1.5) = 1
x1_m = int(np.floor(r/rScale))      # floor(2.5) = 2
x2_m = int(np.ceil(r/rScale))       # ceil(2.5) = 3
x3_m = int(np.ceil(r/rScale + 1))   # ceil(3.5) = 4

print(f"MATLAB indices: x0={x0_m}, x1={x1_m}, x2={x2_m}, x3={x3_m}")
print()

# In MATLAB, inputArray(1) is the FIRST element
# In Python, inputArray[0] is the FIRST element
# So to access the same element:
# MATLAB inputArray(i) = Python inputArray[i-1]

print("MATLAB array access (1-based):")
print(f"  inputArray({x0_m}) = first element  (if x0=1)")
print(f"  inputArray({x1_m}) = second element")
print(f"  inputArray({x2_m}) = third element")
print(f"  inputArray({x3_m}) = fourth element")
print()

print("Python equivalent (0-based):")
print(f"  inputArray[{x0_m-1}] = first element")
print(f"  inputArray[{x1_m-1}] = second element")
print(f"  inputArray[{x2_m-1}] = third element")
print(f"  inputArray[{x3_m-1}] = fourth element")
print()

# Test with actual array
test_array_matlab_style = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
test_array_python = np.array(test_array_matlab_style)

print("Test array: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
print()

print("MATLAB would access:")
print(f"  y0 = inputArray(max({x0_m},1)) = inputArray({max(x0_m,1)}) = {test_array_python[max(x0_m,1)-1]}")
print(f"  y1 = inputArray(max({x1_m},1)) = inputArray({max(x1_m,1)}) = {test_array_python[max(x1_m,1)-1]}")
print(f"  y2 = inputArray(max({x2_m},1)) = inputArray({max(x2_m,1)}) = {test_array_python[max(x2_m,1)-1]}")
print(f"  y3 = inputArray(max({x3_m},1)) = inputArray({max(x3_m,1)}) = {test_array_python[max(x3_m,1)-1]}")
print()

print("Python SHOULD access the SAME elements:")
print(f"  y0 = inputArray[{max(x0_m,1)-1}] = {test_array_python[max(x0_m,1)-1]}")
print(f"  y1 = inputArray[{max(x1_m,1)-1}] = {test_array_python[max(x1_m,1)-1]}")
print(f"  y2 = inputArray[{max(x2_m,1)-1}] = {test_array_python[max(x2_m,1)-1]}")
print(f"  y3 = inputArray[{max(x3_m,1)-1}] = {test_array_python[max(x3_m,1)-1]}")
print()

print("Python CURRENTLY accesses (WRONG):")
print(f"  y0 = inputArray[max({x0_m},0)] = inputArray[{max(x0_m,0)}] = {test_array_python[max(x0_m,0)]}")
print(f"  y1 = inputArray[max({x1_m},0)] = inputArray[{max(x1_m,0)}] = {test_array_python[max(x1_m,0)]}")
print(f"  y2 = inputArray[max({x2_m},0)] = inputArray[{max(x2_m,0)}] = {test_array_python[max(x2_m,0)]}")
print(f"  y3 = inputArray[max({x3_m},0)] = inputArray[{max(x3_m,0)}] = {test_array_python[max(x3_m,0)]}")
print()

print("=" * 80)
print("CONCLUSION:")
print("=" * 80)
print()
print("The Python code is INCORRECT!")
print()
print("MATLAB: y0 = inputArray(max(x0,1))")
print("        - x0 is calculated as an index (1-based)")
print("        - max(x0,1) ensures index is at least 1")
print("        - inputArray(i) accesses the i-th element (1-based)")
print()
print("Python: y0 = input_array[max(x0, 0)]")
print("        - x0 is calculated THE SAME WAY (but result is 1-based MATLAB index!)")
print("        - max(x0, 0) ensures index is at least 0")
print("        - input_array[i] accesses the i-th element (0-based)")
print()
print("FIX: Since x0, x1, x2, x3 are calculated as MATLAB indices,")
print("     we must convert them to Python indices!")
print()
print("CORRECT Python code:")
print("  y0 = input_array[max(x0-1, 0)]  # Convert MATLAB index to Python index")
print("  y1 = input_array[max(x1-1, 0)]")
print("  y2 = input_array[max(x2-1, 0)]")
print("  y3 = input_array[max(x3-1, 0)]")
print()
print("But wait... let me check if Python code recalculates indices differently...")
