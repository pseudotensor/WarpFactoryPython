# Warp Shell Examples

This directory contains examples demonstrating the warp shell metric implementation in WarpFactory.

## Overview

The warp shell is a spherical shell of matter that can create a warp effect for objects within the shell. Unlike the Alcubierre warp drive which uses a warp bubble, the warp shell provides a different approach to achieving faster-than-light effective velocities.

## Key Concepts

### What is a Warp Shell?

A warp shell consists of:
- **Spherical geometry**: Matter distributed in a shell between inner radius R1 and outer radius R2
- **TOV structure**: The Tolman-Oppenheimer-Volkoff equations govern the density and pressure profiles
- **Comoving frame**: The metric is constructed in the frame moving with the warp drive
- **Warp effect**: Controlled spacetime warping inside the shell

### Physical Principles

The warp shell implementation includes:
1. **Density profile**: Constant density matter within the shell boundaries
2. **Pressure profile**: Computed from TOV equations to ensure hydrostatic equilibrium
3. **Mass distribution**: Integrated from the density profile
4. **Metric functions**: Derived from Einstein's field equations
5. **Shift vector**: Creates the warp effect when enabled

## Notebooks

### W1_Warp_Shell.ipynb

**Comprehensive warp shell example showing TOV equations and shell geometry**

This notebook demonstrates:
- Creating a warp shell metric using `get_warp_shell_comoving_metric()`
- Understanding TOV (Tolman-Oppenheimer-Volkoff) equations
- Visualizing physical profiles (density, pressure, mass, metric functions)
- Computing and plotting metric tensor components
- Evaluating stress-energy tensor and energy conditions
- Analyzing exotic matter requirements

**Topics Covered:**
- Shell geometry parameters (R1, R2, mass, velocity)
- TOV equation integration for pressure profiles
- Metric function computation (A and B components)
- Compactness parameter analysis
- Coordinate transformations (spherical to Cartesian)
- Energy density and momentum visualization
- Energy condition violations (Null, Weak, Strong, Dominant)

## Using the Warp Shell Metric

### Basic Usage

```python
from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric

# Define parameters
grid_size = [1, 100, 100, 5]
world_center = [0.5, 50, 50, 2.5]
R1 = 10.0  # Inner radius (m)
R2 = 20.0  # Outer radius (m)
m = 1e30   # Total mass (kg)
v_warp = 0.02  # Velocity as fraction of c

# Create the metric
metric = get_warp_shell_comoving_metric(
    grid_size=grid_size,
    world_center=world_center,
    m=m,
    R1=R1,
    R2=R2,
    v_warp=v_warp,
    do_warp=True
)
```

### Key Parameters

- `grid_size`: World size in [t, x, y, z]
- `world_center`: World center location in [t, x, y, z]
- `m`: Total mass of the warp shell
- `R1`: Inner radius of the shell
- `R2`: Outer radius of the shell
- `Rbuff`: Buffer distance between shell wall and where shift starts (default: 0)
- `sigma`: Sharpness parameter of the shift sigmoid (default: 0)
- `smooth_factor`: Factor by which to smooth the walls of the shell (default: 1)
- `v_warp`: Speed of the warp drive in factors of c (default: 0)
- `do_warp`: Whether to create the warp effect inside the shell (default: False)
- `grid_scaling`: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

### Accessing Physical Profiles

The metric stores computed physical profiles in its parameters:

```python
# Extract profiles
r_vec = metric.params['rVec']  # Radial coordinate array
rho = metric.params['rhoSmooth']  # Smoothed density profile
P = metric.params['PSmooth']  # Smoothed pressure profile
M = metric.params['M']  # Mass profile
A = metric.params['A']  # Metric function A (g_tt)
B = metric.params['B']  # Metric function B (g_rr)
```

## Mathematical Background

### TOV Equations

The Tolman-Oppenheimer-Volkoff equation describes the structure of a spherically symmetric body:

$$\frac{dP}{dr} = -\frac{G(\rho + P/c^2)(m(r) + 4\pi r^3 P/c^2)}{r^2(1 - 2Gm(r)/(rc^2))}$$

where:
- $P(r)$ is the pressure at radius r
- $\rho(r)$ is the density at radius r
- $m(r) = \int_0^r 4\pi r'^2 \rho(r') dr'$ is the mass within radius r

### Metric Form

In spherical coordinates (t, r, θ, φ), the metric has the form:

$$ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2(d\theta^2 + \sin^2\theta d\phi^2)$$

where:
- $A(r) = -e^{2\alpha(r)}$ (temporal component)
- $B(r) = (1 - 2Gm(r)/(rc^2))^{-1}$ (radial component)

The metric is then transformed to Cartesian coordinates for numerical computation.

## References

1. Fell, S. D. B. & Heisenberg, L. (2024). "Constant velocity warp drive". *Classical and Quantum Gravity*, **41**(6). [https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa](https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa)

2. Tolman, R. C. (1939). "Static Solutions of Einstein's Field Equations for Spheres of Fluid". *Physical Review*, **55**(4), 364-373.

3. Oppenheimer, J. R. & Volkoff, G. M. (1939). "On Massive Neutron Cores". *Physical Review*, **55**(4), 374-381.

## Tips and Best Practices

1. **Computational Cost**: The warp shell metric involves integration of TOV equations over a high-resolution radial grid, which can be computationally expensive. Start with smaller grid sizes for testing.

2. **Smoothing Factor**: The `smooth_factor` parameter controls how smoothly the density and pressure transition at the shell boundaries. Higher values (e.g., 4000) produce smoother profiles but may wash out sharp features.

3. **Mass and Compactness**: The total mass `m` affects the compactness parameter $2Gm/(rc^2)$. Values approaching 1 indicate near-black-hole conditions. Stay well below this limit for physical shells.

4. **Warp Velocity**: The `v_warp` parameter should be kept small (e.g., 0.01-0.1) to avoid extreme metric values and numerical instabilities.

5. **Grid Resolution**: Use appropriate `space_scale` to balance resolution and computation time. The shell boundaries should span multiple grid points for accurate representation.

## Related Examples

- **01_metrics/M1_First_Metric.ipynb**: Introduction to metrics
- **01_metrics/M2_Default_Metrics.ipynb**: Overview of available metrics
- **02_energy_tensor/T1_First_Energy_Tensor.ipynb**: Computing energy tensors
- **03_analysis/A1_Energy_Conditions.ipynb**: Energy condition analysis

## Support

For questions or issues with the warp shell examples, please open an issue on the GitHub repository.
