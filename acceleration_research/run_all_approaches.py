"""
Master Runner Script

Runs all 6 acceleration approaches, compares results, and generates comprehensive
analysis. This is the main entry point for the acceleration research.

Usage:
    python run_all_approaches.py [--quick] [--full] [--save-dir DIR]

Options:
    --quick: Run with small grid for quick testing (default)
    --full: Run with full resolution grid (slow, for publication)
    --save-dir: Directory to save results (default: ./results)
"""

import sys
import os
import argparse
import pickle
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all approaches
from acceleration_research.approach1_gradual_transition import run_gradual_transition_simulation
from acceleration_research.approach2_mass_modulation import run_mass_modulation_simulation
from acceleration_research.approach3_hybrid_metrics import run_hybrid_metrics_simulation
from acceleration_research.approach4_multi_shell import run_multi_shell_simulation
from acceleration_research.approach5_modified_lapse import run_modified_lapse_simulation
from acceleration_research.approach6_gw_emission import run_gw_emission_simulation

# Import analysis tools
from acceleration_research.results_comparison import (
    compare_all_approaches,
    print_comparison_report,
    plot_comparison_charts,
    generate_latex_table,
    save_results
)


def run_all_simulations(grid_size=(10, 20, 20, 20), spatial_extent=50.0, verbose=True):
    """
    Run all 6 approaches with consistent parameters

    Args:
        grid_size: Spacetime grid size
        spatial_extent: Spatial domain size in meters
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    all_results = {}

    print("\n" + "="*80)
    print("RUNNING ALL ACCELERATION APPROACHES")
    print("="*80)
    print(f"\nGrid size: {grid_size}")
    print(f"Spatial extent: {spatial_extent} meters")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n")

    # Approach 1: Gradual Transition (Benchmark)
    try:
        print("\n" + "="*80)
        print("APPROACH 1: GRADUAL TRANSITION (Benchmark)")
        print("="*80 + "\n")

        result1 = run_gradual_transition_simulation(
            params=None,  # Use defaults
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 1: Gradual Transition'] = result1
        print("\n✓ Approach 1 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 1 FAILED: {e}\n")
        all_results['Approach 1: Gradual Transition'] = None

    # Approach 3: Hybrid Metrics (HIGHEST PRIORITY)
    try:
        print("\n" + "="*80)
        print("APPROACH 3: HYBRID METRICS (Highest Priority)")
        print("="*80 + "\n")

        result3 = run_hybrid_metrics_simulation(
            params=None,
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 3: Hybrid Metrics'] = result3
        print("\n✓ Approach 3 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 3 FAILED: {e}\n")
        all_results['Approach 3: Hybrid Metrics'] = None

    # Approach 5: Modified Lapse
    try:
        print("\n" + "="*80)
        print("APPROACH 5: MODIFIED LAPSE")
        print("="*80 + "\n")

        result5 = run_modified_lapse_simulation(
            params=None,
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 5: Modified Lapse'] = result5
        print("\n✓ Approach 5 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 5 FAILED: {e}\n")
        all_results['Approach 5: Modified Lapse'] = None

    # Approach 4: Multi-Shell
    try:
        print("\n" + "="*80)
        print("APPROACH 4: MULTI-SHELL")
        print("="*80 + "\n")

        result4 = run_multi_shell_simulation(
            params=None,
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 4: Multi-Shell'] = result4
        print("\n✓ Approach 4 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 4 FAILED: {e}\n")
        all_results['Approach 4: Multi-Shell'] = None

    # Approach 2: Mass Modulation
    try:
        print("\n" + "="*80)
        print("APPROACH 2: MASS MODULATION")
        print("="*80 + "\n")

        result2 = run_mass_modulation_simulation(
            params=None,
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 2: Mass Modulation'] = result2
        print("\n✓ Approach 2 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 2 FAILED: {e}\n")
        all_results['Approach 2: Mass Modulation'] = None

    # Approach 6: GW Emission (Revolutionary)
    try:
        print("\n" + "="*80)
        print("APPROACH 6: GRAVITATIONAL WAVE EMISSION (Revolutionary)")
        print("="*80 + "\n")

        result6 = run_gw_emission_simulation(
            params=None,
            grid_size=grid_size,
            spatial_extent=spatial_extent,
            verbose=verbose
        )
        all_results['Approach 6: GW Emission'] = result6
        print("\n✓ Approach 6 complete\n")
    except Exception as e:
        print(f"\n✗ Approach 6 FAILED: {e}\n")
        all_results['Approach 6: GW Emission'] = None

    print("\n" + "="*80)
    print("ALL SIMULATIONS COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {sum(1 for v in all_results.values() if v is not None)}/{len(all_results)}")

    return all_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run all warp drive acceleration approaches'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with small grid')
    parser.add_argument('--full', action='store_true',
                       help='Full resolution simulation')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'figures'), exist_ok=True)

    # Set grid size based on mode
    if args.full:
        grid_size = (30, 60, 60, 60)
        spatial_extent = 120.0
        print("\nRunning FULL RESOLUTION simulation (this will take a while...)")
    else:
        grid_size = (10, 20, 20, 20)
        spatial_extent = 50.0
        print("\nRunning QUICK TEST simulation")

    # Run all approaches
    all_results = run_all_simulations(
        grid_size=grid_size,
        spatial_extent=spatial_extent,
        verbose=True
    )

    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.save_dir, f'all_results_{timestamp}.pkl')
    save_results(all_results, results_file)
    print(f"\nRaw results saved to: {results_file}")

    # Compare approaches
    print("\n" + "="*80)
    print("ANALYZING AND COMPARING RESULTS")
    print("="*80 + "\n")

    # Filter out failed runs
    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if len(valid_results) == 0:
        print("ERROR: All approaches failed! Cannot perform comparison.")
        return

    comparison = compare_all_approaches(valid_results)

    # Print comparison report
    print_comparison_report(comparison, verbose=True)

    # Save comparison
    comparison_file = os.path.join(args.save_dir, f'comparison_{timestamp}.pkl')
    with open(comparison_file, 'wb') as f:
        pickle.dump(comparison, f)
    print(f"\nComparison saved to: {comparison_file}")

    # Generate plots
    print("\nGenerating comparison plots...")
    figures_dir = os.path.join(args.save_dir, 'figures')
    plot_comparison_charts(comparison, save_dir=figures_dir)

    # Generate LaTeX table
    latex_file = os.path.join(args.save_dir, f'comparison_table_{timestamp}.tex')
    generate_latex_table(comparison, filename=latex_file)

    print("\n" + "="*80)
    print("RESEARCH COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {args.save_dir}")
    print("\nFiles generated:")
    print(f"  - {results_file}")
    print(f"  - {comparison_file}")
    print(f"  - {latex_file}")
    print(f"  - Figures in {figures_dir}/")

    if comparison['overall_rankings']:
        best_approach, best_score = comparison['overall_rankings'][0]
        print(f"\n{'='*80}")
        print(f"BEST APPROACH: {best_approach}")
        print(f"Average score: {best_score:.6e}")
        print(f"{'='*80}\n")

    return all_results, comparison


if __name__ == "__main__":
    main()
