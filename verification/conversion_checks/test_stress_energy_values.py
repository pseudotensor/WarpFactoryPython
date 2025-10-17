#!/usr/bin/env python3
"""
Test to examine actual stress-energy tensor values in Fuchs shell
to determine if ~10^40 magnitudes are physical or numerical errors.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.solver.energy import get_energy_tensor


def analyze_stress_energy_magnitudes():
    """
    Analyze the magnitudes of stress-energy tensor components
    to determine if ~10^40 values are reasonable or indicate bugs.
    """
    print("="*80)
    print("STRESS-ENERGY TENSOR MAGNITUDE ANALYSIS")
    print("="*80)
    print()

    # Paper parameters
    M = 4.49e27  # kg
    R1 = 10.0    # m
    R2 = 20.0    # m

    # Physical constants
    c = 299792458  # m/s
    G = 6.67430e-11  # m^3/(kg*s^2)

    print(f"Physical parameters:")
    print(f"  Mass M = {M:.2e} kg")
    print(f"  Inner radius R1 = {R1} m")
    print(f"  Outer radius R2 = {R2} m")
    print()

    # Expected order of magnitude for energy density
    # ρ ~ M / V ~ M / (4π/3 * R^3)
    volume = (4/3) * np.pi * (R2**3 - R1**3)
    rho_estimate = M / volume
    print(f"Mass shell volume = {volume:.2e} m³")
    print(f"Expected ρ ~ M/V = {rho_estimate:.2e} kg/m³")
    print()

    # In geometric units, stress-energy has units of energy density
    # T_μν has units of [energy / volume] = [kg/(m*s²)]
    T_estimate = rho_estimate * c**2
    print(f"Expected T ~ ρc² = {T_estimate:.2e} kg/(m*s²)")
    print(f"  = {T_estimate:.2e} J/m³")
    print()

    # Create metric (small grid)
    print("Creating metric...")
    grid_size = [1, 11, 11, 11]
    metric = get_warp_shell_comoving_metric(
        grid_size=grid_size,
        world_center=[0.5, 6.0, 6.0, 6.0],
        m=M,
        R1=R1,
        R2=R2,
        v_warp=0.0,
        do_warp=False
    )
    print("✓ Metric created")
    print()

    # Compute stress-energy
    print("Computing stress-energy tensor...")
    energy = get_energy_tensor(metric, try_gpu=False)
    print("✓ Stress-energy computed")
    print()

    # Analyze components
    print("Stress-energy tensor components:")
    print("-" * 80)

    for i in range(4):
        for j in range(4):
            component = energy[(i, j)]

            # Get statistics
            min_val = np.min(component)
            max_val = np.max(component)
            mean_val = np.mean(component[np.isfinite(component)])
            abs_max = np.max(np.abs(component[np.isfinite(component)]))

            # Count NaN/Inf
            n_nan = np.sum(np.isnan(component))
            n_inf = np.sum(np.isinf(component))

            print(f"  T[{i},{j}]:")
            print(f"    Min:     {min_val:+.6e}")
            print(f"    Max:     {max_val:+.6e}")
            print(f"    Mean:    {mean_val:+.6e}")
            print(f"    |Max|:   {abs_max:.6e}")
            print(f"    NaN:     {n_nan}")
            print(f"    Inf:     {n_inf}")

            # Compare to expected
            if abs_max > 0:
                ratio = abs_max / T_estimate
                print(f"    Ratio to expected: {ratio:.2e}")
            print()

    # Check for problematic patterns
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    # Check T_00 (energy density)
    T_00 = energy[(0, 0)]
    T_00_max = np.max(np.abs(T_00[np.isfinite(T_00)]))

    print(f"Energy density T_00:")
    print(f"  Expected magnitude: {T_estimate:.2e}")
    print(f"  Actual magnitude:   {T_00_max:.2e}")
    print(f"  Ratio:              {T_00_max / T_estimate:.2e}")
    print()

    if T_00_max / T_estimate > 1e10:
        print("⚠ WARNING: Stress-energy is ~10^{} times larger than expected!".format(
            int(np.log10(T_00_max / T_estimate))
        ))
        print("  This suggests:")
        print("  1. Numerical overflow in stress-energy calculation")
        print("  2. Incorrect units or scaling factors")
        print("  3. Bug in Einstein tensor calculation")
        print("  4. Singularity in metric near shell boundaries")
    elif T_00_max / T_estimate < 1e-10:
        print("⚠ WARNING: Stress-energy is ~10^{} times smaller than expected!".format(
            -int(np.log10(T_00_max / T_estimate))
        ))
        print("  This suggests numerical underflow or incorrect calculation")
    else:
        print("✓ Stress-energy magnitude is within reasonable range")

    print()
    print("="*80)

    # Look at specific spatial locations
    print()
    print("Spatial distribution of T_00:")
    print("-" * 80)

    # Center slice
    T_00_slice = T_00[0, grid_size[1]//2, grid_size[2]//2, :]

    print(f"Along z-axis through center:")
    for k in range(grid_size[3]):
        val = T_00_slice[k]
        print(f"  z={k}: T_00 = {val:+.6e}")

    print()
    print("="*80)

    return T_00_max / T_estimate


if __name__ == "__main__":
    print("\n")
    print("*"*80)
    print("Investigating if ~10^40 values are numerical bugs or physics")
    print("*"*80)
    print()

    ratio = analyze_stress_energy_magnitudes()

    print()
    print("*"*80)
    print("CONCLUSION")
    print("*"*80)
    print()

    if abs(np.log10(ratio)) < 2:
        print("✓ Stress-energy magnitudes are physically reasonable")
        print("  The ~10^40 violations may be due to:")
        print("  - Actual violation of energy conditions by this geometry")
        print("  - Subtle bugs in energy condition evaluation code")
    else:
        print(f"✗ Stress-energy magnitudes are off by factor ~10^{int(np.log10(ratio))}")
        print("  This indicates a BUG in:")
        print("  - Stress-energy calculation (get_energy_tensor)")
        print("  - Metric calculation")
        print("  - Units/scaling conversion")
        print()
        print("  The ~10^40 energy condition violations are likely artifacts")
        print("  of incorrect stress-energy tensor computation, not the")
        print("  energy condition evaluation code itself.")

    print()
    print("*"*80)
    print()
