"""
Parameter Space Exploration

Systematically explores parameter space for each approach to find optimal
configurations and understand sensitivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple
import sys
import os
from itertools import product
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def explore_parameter_1d(
    run_function: Callable,
    param_name: str,
    param_values: List,
    base_params: Dict,
    grid_size: Tuple = (10, 20, 20, 20),
    verbose: bool = True
) -> Dict:
    """
    Explore single parameter

    Args:
        run_function: Function that runs simulation
        param_name: Name of parameter to vary
        param_values: List of values to test
        base_params: Base parameter dictionary
        grid_size: Grid size for simulations
        verbose: Print progress

    Returns:
        Dictionary of results for each parameter value
    """
    results = {}

    for i, value in enumerate(param_values):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Testing {param_name} = {value} ({i+1}/{len(param_values)})")
            print(f"{'='*70}\n")

        # Update parameter
        test_params = base_params.copy()
        test_params[param_name] = value

        # Run simulation
        try:
            result = run_function(
                params=test_params,
                grid_size=grid_size,
                spatial_extent=80.0,
                verbose=verbose
            )
            results[value] = result
        except Exception as e:
            if verbose:
                print(f"ERROR: Simulation failed for {param_name}={value}: {e}")
            results[value] = None

    return results


def explore_parameter_2d(
    run_function: Callable,
    param1_name: str,
    param1_values: List,
    param2_name: str,
    param2_values: List,
    base_params: Dict,
    grid_size: Tuple = (10, 20, 20, 20),
    verbose: bool = True
) -> Dict:
    """
    Explore two parameters simultaneously

    Args:
        run_function: Function that runs simulation
        param1_name: First parameter name
        param1_values: First parameter values
        param2_name: Second parameter name
        param2_values: Second parameter values
        base_params: Base parameter dictionary
        grid_size: Grid size
        verbose: Print progress

    Returns:
        Dictionary with results for each parameter combination
    """
    results = {}
    total = len(param1_values) * len(param2_values)
    count = 0

    for val1 in param1_values:
        for val2 in param2_values:
            count += 1
            if verbose:
                print(f"\n{'='*70}")
                print(f"Test {count}/{total}: {param1_name}={val1}, {param2_name}={val2}")
                print(f"{'='*70}\n")

            # Update parameters
            test_params = base_params.copy()
            test_params[param1_name] = val1
            test_params[param2_name] = val2

            # Run simulation
            try:
                result = run_function(
                    params=test_params,
                    grid_size=grid_size,
                    spatial_extent=80.0,
                    verbose=False  # Less verbose for 2D scans
                )
                results[(val1, val2)] = result
            except Exception as e:
                if verbose:
                    print(f"ERROR: Failed: {e}")
                results[(val1, val2)] = None

    return results


def plot_1d_parameter_scan(
    results: Dict,
    param_name: str,
    condition: str = "Null",
    save_path: str = None
):
    """Plot results of 1D parameter scan"""
    param_values = list(results.keys())
    worst_violations = []
    l2_norms = []

    for value in param_values:
        if results[value] is None:
            worst_violations.append(np.nan)
            l2_norms.append(np.nan)
        else:
            metrics = results[value]['metrics'][condition]
            worst_violations.append(metrics['worst_violation'])
            l2_norms.append(metrics['total_violation_L2'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Worst violation
    ax1.plot(param_values, worst_violations, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Worst Violation')
    ax1.set_title(f'{condition} Energy Condition: Worst Violation')
    ax1.grid(alpha=0.3)

    # L2 norm
    ax2.plot(param_values, l2_norms, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('L2 Norm')
    ax2.set_title(f'{condition} Energy Condition: Total Violation (L2)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def plot_2d_parameter_scan(
    results: Dict,
    param1_name: str,
    param1_values: List,
    param2_name: str,
    param2_values: List,
    condition: str = "Null",
    save_path: str = None
):
    """Plot results of 2D parameter scan as heatmap"""
    # Create grid
    Z = np.zeros((len(param2_values), len(param1_values)))

    for i, val2 in enumerate(param2_values):
        for j, val1 in enumerate(param1_values):
            if (val1, val2) in results and results[(val1, val2)] is not None:
                metrics = results[(val1, val2)]['metrics'][condition]
                Z[i, j] = metrics['worst_violation']
            else:
                Z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='RdYlGn',
                   extent=[param1_values[0], param1_values[-1],
                          param2_values[0], param2_values[-1]])

    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_title(f'{condition} Energy Condition: Worst Violation\n'
                f'(Green = Better, Red = Worse)')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Worst Violation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


def find_optimal_parameters(
    results: Dict,
    condition: str = "Null"
) -> Tuple:
    """
    Find parameter values that minimize violations

    Args:
        results: Results dictionary from parameter scan
        condition: Energy condition to optimize

    Returns:
        (best_params, best_score, best_result)
    """
    best_score = -np.inf
    best_params = None
    best_result = None

    for params, result in results.items():
        if result is None:
            continue

        metrics = result['metrics'][condition]
        score = metrics['worst_violation']

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_params = params
            best_result = result

    return best_params, best_score, best_result


def comprehensive_parameter_study(
    approach_name: str,
    run_function: Callable,
    parameter_ranges: Dict[str, List],
    base_params: Dict,
    save_dir: str = "./results"
) -> Dict:
    """
    Comprehensive parameter study for an approach

    Args:
        approach_name: Name of approach
        run_function: Simulation function
        parameter_ranges: Dict mapping parameter names to value lists
        base_params: Base parameters
        save_dir: Directory to save results

    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE PARAMETER STUDY: {approach_name}")
    print("=" * 80)

    study_results = {
        'approach': approach_name,
        'base_params': base_params,
        'parameter_ranges': parameter_ranges,
        '1d_scans': {},
        'optimal_params': {}
    }

    # 1D scans for each parameter
    for param_name, param_values in parameter_ranges.items():
        print(f"\n{'='*80}")
        print(f"1D SCAN: {param_name}")
        print(f"{'='*80}")

        results_1d = explore_parameter_1d(
            run_function=run_function,
            param_name=param_name,
            param_values=param_values,
            base_params=base_params,
            grid_size=(10, 20, 20, 20),
            verbose=True
        )

        study_results['1d_scans'][param_name] = results_1d

        # Find optimal for this parameter
        for condition in ['Null', 'Weak', 'Dominant', 'Strong']:
            best_val, best_score, best_result = find_optimal_parameters(
                results_1d, condition
            )

            if param_name not in study_results['optimal_params']:
                study_results['optimal_params'][param_name] = {}

            study_results['optimal_params'][param_name][condition] = {
                'value': best_val,
                'score': best_score
            }

        # Plot
        plot_path = os.path.join(save_dir, f"{approach_name}_{param_name}_scan.png")
        plot_1d_parameter_scan(results_1d, param_name, condition="Null",
                              save_path=plot_path)

    # Save results
    save_path = os.path.join(save_dir, f"{approach_name}_parameter_study.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(study_results, f)
    print(f"\nResults saved to: {save_path}")

    return study_results


if __name__ == "__main__":
    print("Parameter Space Exploration Tool")
    print("=" * 70)
    print("\nThis module provides systematic parameter space exploration.")
    print("\nUsage:")
    print("  from parameter_space_exploration import explore_parameter_1d")
    print("  results = explore_parameter_1d(run_function, 'tau', [10, 25, 50, 100], base_params)")
    print("  plot_1d_parameter_scan(results, 'tau')")
