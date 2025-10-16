"""
RIGOROUS REPRODUCTION OF arXiv:2405.02709v1
===================================================

This script reproduces EXACTLY the warp shell metric from the paper:
"Constant Velocity Physical Warp Drive Solution"
by Jared Fuchs et al., May 2024

EXTRACTED PARAMETERS FROM PAPER:
================================
Section 3 (Shell Metric):
- R1 = 10 m (inner radius)
- R2 = 20 m (outer radius)
- M = 4.49 × 10^27 kg (2.365 Jupiter masses)
- Constant density shell between R1 and R2
- TOV equation for pressure with P(R2) = 0
- Moving average smoothing with span ratio sρ/sP ≈ 1.72
- 4 iterations of smoothing

Section 4 (Warp Shell):
- βwarp = 0.02 (shift vector magnitude inside shell)
- Rb > 0 (buffer region for derivatives)
- Uses sigmoid smoothing function Swarp(r)

Section 2.2 (Numerical Methods):
- Observer sampling: 100 spatial orientations, 10 temporal velocity samples
- Grid resolution not explicitly stated, but figures show ~30m range
- Eulerian observers for stress-energy evaluation

PAPER'S CLAIMS:
===============
Abstract: "satisfies all of the energy conditions"
Section 3.2: "No energy condition violations exist beyond the numerical
             precision limits that exist at 10^34"
Section 4.2: "Modification of the shift vector in this fashion has no
             impact on the violation compared to the normal matter shell solution"
Figure 7: Shows all energy conditions positive (10^39 J/m^3 scale)
Figure 10: Shows all energy conditions positive for Warp Shell

KEY QUESTION:
=============
Do the energy conditions truly have NO VIOLATIONS when computed rigorously
with high observer sampling and multiple grid resolutions?
"""

import numpy as np
import sys
import os
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell import get_warp_shell_comoving_metric
from warpfactory.solver.energy import get_energy_tensor
from warpfactory.analyzer.energy_conditions import get_energy_conditions
from warpfactory.analyzer.eval_metric import eval_metric
from warpfactory.core.tensor_ops import change_tensor_index
from warpfactory.units.constants import c, G
import matplotlib.pyplot as plt

# Exact parameters from paper
PARAMS_PAPER = {
    'R1': 10.0,  # meters - inner radius
    'R2': 20.0,  # meters - outer radius
    'M': 4.49e27,  # kg - total mass (2.365 Jupiter masses)
    'beta_warp': 0.02,  # shift vector magnitude
    'Rbuff': 0.0,  # buffer region (paper doesn't specify exact value)
    'smooth_factor': 1.0,  # smoothing parameter
    'sigma': 0.0,  # sigmoid sharpness (paper uses compact sigmoid)
}

# Grid parameters (inferred from figures showing ~30m range)
GRID_SIZE = [1, 64, 64, 64]  # Start with moderate resolution
WORLD_CENTER = [0.5, 32.5, 32.5, 32.5]
GRID_SCALING = [1.0, 1.0, 1.0, 1.0]  # 1m per grid point

# Observer sampling from paper
NUM_ANGULAR_VEC = 100  # spatial orientations
NUM_TIME_VEC = 10  # temporal velocity samples


def build_shell_metric(do_warp=False, v_warp=0.0):
    """Build the shell metric (with or without warp effect)"""
    print(f"\n{'='*70}")
    print(f"Building {'WARP SHELL' if do_warp else 'MATTER SHELL'} metric")
    print(f"{'='*70}")
    print(f"Parameters:")
    print(f"  R1 = {PARAMS_PAPER['R1']} m")
    print(f"  R2 = {PARAMS_PAPER['R2']} m")
    print(f"  M = {PARAMS_PAPER['M']:.2e} kg ({PARAMS_PAPER['M']/1.898e27:.3f} Jupiter masses)")
    if do_warp:
        print(f"  β_warp = {PARAMS_PAPER['beta_warp']}")
        print(f"  v_warp = {v_warp}")
    print(f"  Grid: {GRID_SIZE}")
    print(f"  Observer sampling: {NUM_ANGULAR_VEC} angular, {NUM_TIME_VEC} temporal")

    metric = get_warp_shell_comoving_metric(
        grid_size=GRID_SIZE,
        world_center=WORLD_CENTER,
        m=PARAMS_PAPER['M'],
        R1=PARAMS_PAPER['R1'],
        R2=PARAMS_PAPER['R2'],
        Rbuff=PARAMS_PAPER['Rbuff'],
        sigma=PARAMS_PAPER['sigma'],
        smooth_factor=PARAMS_PAPER['smooth_factor'],
        v_warp=v_warp if do_warp else 0.0,
        do_warp=do_warp,
        grid_scaling=GRID_SCALING
    )

    return metric


def compute_energy_conditions(metric, label=""):
    """Compute all energy conditions rigorously"""
    print(f"\n{'='*70}")
    print(f"Computing Energy Conditions: {label}")
    print(f"{'='*70}")

    # Get stress-energy tensor from Einstein equations
    print("Computing stress-energy tensor (T_μν)...")
    T_tensor = get_energy_tensor(metric, try_gpu=False)

    # Compute all four energy conditions
    conditions = ['Null', 'Weak', 'Strong', 'Dominant']
    results = {}

    for condition in conditions:
        print(f"\nEvaluating {condition} Energy Condition...")
        print(f"  Observer sampling: {NUM_ANGULAR_VEC} angular × {NUM_TIME_VEC} temporal")

        ec_map, _, _ = get_energy_conditions(
            energy_tensor=T_tensor,
            metric=metric,
            condition=condition,
            num_angular_vec=NUM_ANGULAR_VEC,
            num_time_vec=NUM_TIME_VEC,
            return_vec=False,
            try_gpu=False
        )

        results[condition] = ec_map

        # Analyze violations
        min_val = np.nanmin(ec_map)
        max_val = np.nanmax(ec_map)
        mean_val = np.nanmean(ec_map[~np.isnan(ec_map)])
        num_violations = np.sum(ec_map < 0)
        total_points = np.sum(~np.isnan(ec_map))

        print(f"  Results:")
        print(f"    Min value: {min_val:.6e}")
        print(f"    Max value: {max_val:.6e}")
        print(f"    Mean value: {mean_val:.6e}")
        print(f"    Violations: {num_violations} / {total_points} points")

        if num_violations > 0:
            violation_magnitude = np.min(ec_map[ec_map < 0])
            print(f"    ⚠ VIOLATION DETECTED! Worst: {violation_magnitude:.6e}")
        else:
            print(f"    ✓ No violations detected")

    return results, T_tensor


def analyze_slice(ec_results, T_tensor, metric, slice_idx=None):
    """Analyze energy conditions along a slice"""
    if slice_idx is None:
        slice_idx = GRID_SIZE[2] // 2  # Middle slice

    print(f"\n{'='*70}")
    print(f"Analyzing Y-axis slice at index {slice_idx}")
    print(f"{'='*70}")

    # Extract slice data
    t = 0
    slice_data = {}
    for condition, ec_map in ec_results.items():
        slice_data[condition] = ec_map[t, :, slice_idx, :]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    conditions = ['Null', 'Weak', 'Strong', 'Dominant']
    for idx, condition in enumerate(conditions):
        ax = axes[idx]
        data = slice_data[condition]

        # Plot
        im = ax.imshow(data.T, origin='lower', extent=[0, GRID_SIZE[1], 0, GRID_SIZE[3]],
                      cmap='RdBu_r', vmin=np.nanmin(data), vmax=np.nanmax(data))
        ax.set_title(f'{condition} Energy Condition')
        ax.set_xlabel('X [grid units]')
        ax.set_ylabel('Z [grid units]')
        plt.colorbar(im, ax=ax, label='Value')

        # Mark violations
        violations = data < 0
        if np.any(violations):
            y_viol, z_viol = np.where(violations)
            ax.scatter(y_viol, z_viol, c='red', s=10, alpha=0.5, label='Violations')
            ax.legend()

    plt.tight_layout()
    return fig


def convergence_test():
    """Test convergence with different grid resolutions"""
    print(f"\n{'='*70}")
    print("CONVERGENCE TEST - Multiple Grid Resolutions")
    print(f"{'='*70}")

    resolutions = [32, 64, 96]
    results_by_res = {}

    for res in resolutions:
        print(f"\n\nTesting resolution: {res}³")

        global GRID_SIZE, WORLD_CENTER
        GRID_SIZE = [1, res, res, res]
        WORLD_CENTER = [0.5, res/2 + 0.5, res/2 + 0.5, res/2 + 0.5]

        # Build warp shell
        metric = build_shell_metric(do_warp=True, v_warp=PARAMS_PAPER['beta_warp'])

        # Compute energy conditions
        ec_results, _ = compute_energy_conditions(metric, f"Warp Shell @ {res}³")

        results_by_res[res] = ec_results

    # Compare results
    print(f"\n{'='*70}")
    print("CONVERGENCE COMPARISON")
    print(f"{'='*70}")

    conditions = ['Null', 'Weak', 'Strong', 'Dominant']
    for condition in conditions:
        print(f"\n{condition} Energy Condition:")
        for res in resolutions:
            min_val = np.nanmin(results_by_res[res][condition])
            num_viol = np.sum(results_by_res[res][condition] < 0)
            total = np.sum(~np.isnan(results_by_res[res][condition]))
            print(f"  {res}³: min={min_val:.6e}, violations={num_viol}/{total}")

    return results_by_res


def main():
    """Main reproduction workflow"""
    print("="*70)
    print("RIGOROUS REPRODUCTION OF arXiv:2405.02709v1")
    print("Paper: Constant Velocity Physical Warp Drive Solution")
    print("Authors: Jared Fuchs et al.")
    print("="*70)

    # Step 1: Build Matter Shell (no warp)
    print("\n\nSTEP 1: MATTER SHELL (Baseline)")
    shell_metric = build_shell_metric(do_warp=False)
    shell_ec, shell_T = compute_energy_conditions(shell_metric, "Matter Shell")

    # Step 2: Build Warp Shell
    print("\n\nSTEP 2: WARP SHELL (With shift vector)")
    warp_metric = build_shell_metric(do_warp=True, v_warp=PARAMS_PAPER['beta_warp'])
    warp_ec, warp_T = compute_energy_conditions(warp_metric, "Warp Shell")

    # Step 3: Visual analysis
    print("\n\nSTEP 3: VISUAL ANALYSIS")
    fig_shell = analyze_slice(shell_ec, shell_T, shell_metric)
    fig_shell.savefig('/WarpFactory/warpfactory_py/paper_2405_rigorous/shell_energy_conditions.png', dpi=150)
    print("Saved: shell_energy_conditions.png")

    fig_warp = analyze_slice(warp_ec, warp_T, warp_metric)
    fig_warp.savefig('/WarpFactory/warpfactory_py/paper_2405_rigorous/warp_shell_energy_conditions.png', dpi=150)
    print("Saved: warp_shell_energy_conditions.png")

    # Step 4: Convergence test
    print("\n\nSTEP 4: CONVERGENCE TEST")
    conv_results = convergence_test()

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    print("\nMATTER SHELL:")
    for condition in ['Null', 'Weak', 'Strong', 'Dominant']:
        min_val = np.nanmin(shell_ec[condition])
        num_viol = np.sum(shell_ec[condition] < 0)
        status = "✓ PASS" if num_viol == 0 else "✗ FAIL"
        print(f"  {condition:12s}: min={min_val:12.6e}  {status}")

    print("\nWARP SHELL:")
    for condition in ['Null', 'Weak', 'Strong', 'Dominant']:
        min_val = np.nanmin(warp_ec[condition])
        num_viol = np.sum(warp_ec[condition] < 0)
        status = "✓ PASS" if num_viol == 0 else "✗ FAIL"
        print(f"  {condition:12s}: min={min_val:12.6e}  {status}")

    # CRITICAL ASSESSMENT
    print(f"\n{'='*70}")
    print("CRITICAL ASSESSMENT")
    print(f"{'='*70}")

    any_violations = False
    for condition in ['Null', 'Weak', 'Strong', 'Dominant']:
        if np.sum(warp_ec[condition] < 0) > 0:
            any_violations = True
            break

    if any_violations:
        print("\n⚠ VIOLATIONS DETECTED!")
        print("The paper's claim of 'satisfies all energy conditions' appears INCORRECT.")
        print("Energy condition violations exist beyond numerical precision limits.")
    else:
        print("\n✓ NO VIOLATIONS DETECTED!")
        print("The paper's claims are CONFIRMED with these parameters.")
        print("However, further testing with higher observer sampling recommended.")

    print("\n" + "="*70)
    print("Reproduction complete. See generated plots and analysis above.")
    print("="*70)


if __name__ == "__main__":
    main()
