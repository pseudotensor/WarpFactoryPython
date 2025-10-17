"""
Debug the _flip_index function
"""
import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.core.tensor import Tensor
from warpfactory.core.tensor_ops import change_tensor_index, _flip_index

def create_minkowski_metric(covariant=True):
    """Create Minkowski metric tensor g_μν = diag(-1, 1, 1, 1)"""
    tensor_dict = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                if i == 0:
                    tensor_dict[(i, j)] = np.array([[[[-1.0]]]])
                else:
                    tensor_dict[(i, j)] = np.array([[[[1.0]]]])
            else:
                tensor_dict[(i, j)] = np.array([[[[0.0]]]])

    index_type = "covariant" if covariant else "contravariant"
    return Tensor(tensor=tensor_dict, tensor_type="metric", coords="cartesian", index=index_type)

# Create Minkowski metric
g_lower = create_minkowski_metric(covariant=True)

print("Minkowski metric g_μν:")
for i in range(4):
    for j in range(4):
        print(f"g[{i},{j}] = {g_lower.tensor[(i,j)][0,0,0,0]:6.1f}", end="  ")
    print()

# Create simple test vector V^μ = (1, 0, 0, 0)
v_up_dict = {}
for i in range(4):
    for j in range(4):
        if i == 0 and j == 0:
            v_up_dict[(i, j)] = np.array([[[[1.0]]]])
        else:
            v_up_dict[(i, j)] = np.array([[[[0.0]]]])

v_contravariant = Tensor(tensor=v_up_dict, tensor_type="stress-energy",
                         coords="cartesian", index="contravariant")

print("\nInput tensor V^μν (contravariant):")
print(f"V^{{0,0}} = {v_contravariant.tensor[(0,0)][0,0,0,0]}")
for i in range(1, 4):
    print(f"V^{{{i},{i}}} = {v_contravariant.tensor[(i,i)][0,0,0,0]}")

# Manual calculation: V_μν = g_μα g_νβ V^αβ
# For V^{00} = 1, all else 0:
# V_{00} = g_{0α} g_{0β} V^{αβ} = g_{00} g_{00} V^{00} = (-1)(-1)(1) = 1
# But we want V_0 from V^0, which is different!

print("\n" + "="*60)
print("ISSUE IDENTIFIED:")
print("="*60)
print("We're treating a rank-1 tensor (vector) as a rank-2 tensor!")
print("V^μ should be stored differently than V^{μν}")
print()
print("For lowering V^μ to V_μ:")
print("  V_μ = g_μν V^ν")
print()
print("But _flip_index does:")
print("  T'_{ij} = g_{ia} g_{jb} T^{ab}")
print()
print("This is for rank-2 tensors, not vectors!")
print("="*60)

# Let's verify what MATLAB does
print("\nChecking MATLAB code structure...")
print("MATLAB line 123: tempOutputTensor{i, j} += inputTensor.tensor{a, b} .* metricTensor.tensor{a, i} .* metricTensor.tensor{b, j}")
print()
print("This is CORRECT for rank-2 tensors.")
print("For T^{μν} -> T_{μν}: T_{ij} = g_{iμ} g_{jν} T^{μν}")
print()
print("The issue is our test case:")
print("We should test with a full rank-2 tensor, not misuse it as a vector.")

# Let's test properly with rank-2 tensor
print("\n" + "="*60)
print("PROPER TEST: Rank-2 tensor")
print("="*60)

# Create T^{μν} = diag(2, 3, 4, 5)
t_up_dict = {}
for i in range(4):
    for j in range(4):
        if i == j:
            t_up_dict[(i, j)] = np.array([[[[float(i+2)]]]])
        else:
            t_up_dict[(i, j)] = np.array([[[[0.0]]]])

t_contravariant = Tensor(tensor=t_up_dict, tensor_type="stress-energy",
                         coords="cartesian", index="contravariant")

print("T^{μν} diagonal:", [t_contravariant.tensor[(i,i)][0,0,0,0] for i in range(4)])

# Lower both indices
t_covariant = change_tensor_index(t_contravariant, "covariant", g_lower)

print("T_{μν} diagonal:", [t_covariant.tensor[(i,i)][0,0,0,0] for i in range(4)])

# Expected: T_{00} = g_{00}^2 * T^{00} = (-1)^2 * 2 = 2
#           T_{11} = g_{11}^2 * T^{11} = (1)^2 * 3 = 3
print("\nExpected T_{μν} diagonal: [2, 3, 4, 5]")

matches = all(np.isclose(t_covariant.tensor[(i,i)][0,0,0,0], float(i+2)) for i in range(4))
if matches:
    print("✓ CORRECT: Rank-2 tensor transformation works!")
else:
    print("✗ ERROR: Rank-2 tensor transformation failed!")

# Now test with off-diagonal element
print("\n" + "="*60)
print("TEST: Off-diagonal element T^{01} = 1")
print("="*60)

t2_dict = {}
for i in range(4):
    for j in range(4):
        if i == 0 and j == 1:
            t2_dict[(i, j)] = np.array([[[[1.0]]]])
        else:
            t2_dict[(i, j)] = np.array([[[[0.0]]]])

t2_contravariant = Tensor(tensor=t2_dict, tensor_type="stress-energy",
                          coords="cartesian", index="contravariant")

t2_covariant = change_tensor_index(t2_contravariant, "covariant", g_lower)

print(f"T^{{01}} = {t2_contravariant.tensor[(0,1)][0,0,0,0]}")
print(f"T_{{01}} = {t2_covariant.tensor[(0,1)][0,0,0,0]}")
print(f"Expected: T_{{01}} = g_{{00}} g_{{11}} T^{{01}} = (-1)(1)(1) = -1")

if np.isclose(t2_covariant.tensor[(0,1)][0,0,0,0], -1.0):
    print("✓ CORRECT!")
else:
    print(f"✗ ERROR: Got {t2_covariant.tensor[(0,1)][0,0,0,0]}, expected -1")
