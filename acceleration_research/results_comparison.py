"""
Results Comparison Tool

This module provides comprehensive comparison and analysis of all acceleration approaches.
Generates quantitative metrics, rankings, and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os
import pickle
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filepath: str) -> Dict:
    """Load saved results from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict, filepath: str):
    """Save results to pickle file (excluding unpicklable objects)"""
    # Create a clean copy without unpicklable objects
    clean_results = {}
    for key, val in results.items():
        if val is None:
            clean_results[key] = None
        else:
            clean_results[key] = {
                'approach': val.get('approach'),
                'params': val.get('params'),
                'results': val.get('results'),
                'metrics': val.get('metrics')
                # Exclude 'warp_drive' and 'time_dependent_metric' which contain lambda functions
            }

    with open(filepath, 'wb') as f:
        pickle.dump(clean_results, f)


def compare_all_approaches(results_dict: Dict[str, Dict]) -> Dict:
    """
    Comprehensive comparison of all approaches

    Args:
        results_dict: Dictionary mapping approach names to their results

    Returns:
        Comparison metrics and rankings
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'approaches': list(results_dict.keys()),
        'conditions': ['Null', 'Weak', 'Dominant', 'Strong'],
        'metrics_by_condition': {},
        'overall_rankings': {}
    }

    # Compare each energy condition
    for condition in comparison['conditions']:
        comparison['metrics_by_condition'][condition] = {}

        for approach_name, results in results_dict.items():
            if 'metrics' not in results or condition not in results['metrics']:
                continue

            metrics = results['metrics'][condition]
            comparison['metrics_by_condition'][condition][approach_name] = {
                'worst_violation': metrics.get('worst_violation', np.nan),
                'max_magnitude': metrics.get('max_magnitude', np.nan),
                'total_L2': metrics.get('total_violation_L2', np.nan),
                'fraction_violating': metrics.get('fraction_violating', np.nan),
                'temporal_extent': metrics.get('temporal_extent', None),
                'peak_time': metrics.get('peak_time', None)
            }

        # Rank approaches for this condition (less negative is better)
        rankings = []
        for approach in comparison['approaches']:
            if approach in comparison['metrics_by_condition'][condition]:
                worst = comparison['metrics_by_condition'][condition][approach]['worst_violation']
                rankings.append((approach, worst))

        rankings.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        comparison['metrics_by_condition'][condition]['rankings'] = rankings

    # Overall ranking (average across conditions)
    overall_scores = {}
    for approach in comparison['approaches']:
        scores = []
        for condition in comparison['conditions']:
            if approach in comparison['metrics_by_condition'][condition]:
                worst = comparison['metrics_by_condition'][condition][approach]['worst_violation']
                if not np.isnan(worst):
                    scores.append(worst)

        if scores:
            overall_scores[approach] = np.mean(scores)
        else:
            overall_scores[approach] = -np.inf

    overall_rankings = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    comparison['overall_rankings'] = overall_rankings

    return comparison


def print_comparison_report(comparison: Dict, verbose: bool = True):
    """Print formatted comparison report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON OF ALL APPROACHES")
    print("=" * 80)
    print(f"\nTimestamp: {comparison['timestamp']}")
    print(f"Approaches compared: {len(comparison['approaches'])}")

    # Overall rankings
    print("\n" + "=" * 80)
    print("OVERALL RANKINGS (Best to Worst)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Approach':<40} {'Avg Score':>15}")
    print("-" * 80)

    for rank, (approach, score) in enumerate(comparison['overall_rankings'], 1):
        print(f"{rank:<6} {approach:<40} {score:>15.6e}")

    # Detailed by condition
    if verbose:
        for condition in comparison['conditions']:
            print("\n" + "=" * 80)
            print(f"{condition.upper()} ENERGY CONDITION COMPARISON")
            print("=" * 80)

            print(f"\n{'Approach':<30} {'Worst Viol':>15} {'Max Mag':>15} {'L2 Norm':>15} {'Frac %':>10}")
            print("-" * 80)

            for approach in comparison['approaches']:
                if approach in comparison['metrics_by_condition'][condition]:
                    data = comparison['metrics_by_condition'][condition][approach]
                    print(f"{approach:<30} {data['worst_violation']:>15.6e} "
                          f"{data['max_magnitude']:>15.6e} {data['total_L2']:>15.6e} "
                          f"{data['fraction_violating']*100:>9.1f}%")

            print(f"\nRankings for {condition}:")
            for rank, (approach, score) in enumerate(
                comparison['metrics_by_condition'][condition]['rankings'], 1
            ):
                print(f"  {rank}. {approach}: {score:.6e}")

    # Best approach identification
    print("\n" + "=" * 80)
    print("BEST APPROACH IDENTIFICATION")
    print("=" * 80)

    if comparison['overall_rankings']:
        best_approach, best_score = comparison['overall_rankings'][0]
        print(f"\nOverall Winner: {best_approach}")
        print(f"Average violation score: {best_score:.6e}")

        print("\nPerformance by condition:")
        for condition in comparison['conditions']:
            if best_approach in comparison['metrics_by_condition'][condition]:
                data = comparison['metrics_by_condition'][condition][best_approach]
                print(f"  {condition}:")
                print(f"    Worst violation: {data['worst_violation']:.6e}")
                print(f"    Fraction violating: {data['fraction_violating']:.1%}")


def plot_comparison_charts(comparison: Dict, save_dir: str = None):
    """Generate comparison plots"""
    approaches = comparison['approaches']
    conditions = comparison['conditions']

    # Figure 1: Worst violations by approach and condition
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(approaches))
    width = 0.2

    for i, condition in enumerate(conditions):
        violations = []
        for approach in approaches:
            if approach in comparison['metrics_by_condition'][condition]:
                viol = comparison['metrics_by_condition'][condition][approach]['worst_violation']
                violations.append(viol if not np.isnan(viol) else 0)
            else:
                violations.append(0)

        ax.bar(x + i*width, violations, width, label=condition)

    ax.set_xlabel('Approach')
    ax.set_ylabel('Worst Violation')
    ax.set_title('Energy Condition Violations by Approach')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([a.split('(')[0].strip() for a in approaches], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_violations.png'), dpi=150)
        print(f"Saved: comparison_violations.png")

    plt.show()

    # Figure 2: Fraction violating
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, condition in enumerate(conditions):
        fractions = []
        for approach in approaches:
            if approach in comparison['metrics_by_condition'][condition]:
                frac = comparison['metrics_by_condition'][condition][approach]['fraction_violating']
                fractions.append(frac * 100 if not np.isnan(frac) else 0)
            else:
                fractions.append(0)

        ax.bar(x + i*width, fractions, width, label=condition)

    ax.set_xlabel('Approach')
    ax.set_ylabel('Fraction Violating (%)')
    ax.set_title('Temporal Extent of Violations')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([a.split('(')[0].strip() for a in approaches], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_fraction.png'), dpi=150)
        print(f"Saved: comparison_fraction.png")

    plt.show()

    # Figure 3: Overall ranking
    fig, ax = plt.subplots(figsize=(10, 6))

    ranks = [r[0] for r in comparison['overall_rankings']]
    scores = [r[1] for r in comparison['overall_rankings']]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ranks)))
    ax.barh(ranks, scores, color=colors)

    ax.set_xlabel('Average Violation Score')
    ax.set_ylabel('Approach')
    ax.set_title('Overall Approach Rankings (Higher is Better)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparison_rankings.png'), dpi=150)
        print(f"Saved: comparison_rankings.png")

    plt.show()


def generate_latex_table(comparison: Dict, filename: str = None):
    """Generate LaTeX table of results"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of Warp Drive Acceleration Approaches}")
    latex.append("\\label{tab:acceleration_comparison}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Approach & NEC & WEC & DEC & SEC \\\\")
    latex.append("\\hline")

    for approach in comparison['approaches']:
        row = [approach.split('(')[0].strip()]
        for condition in comparison['conditions']:
            if approach in comparison['metrics_by_condition'][condition]:
                viol = comparison['metrics_by_condition'][condition][approach]['worst_violation']
                row.append(f"{viol:.2e}")
            else:
                row.append("---")

        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    if filename:
        with open(filename, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {filename}")

    return latex_str


if __name__ == "__main__":
    print("Results Comparison Tool")
    print("=" * 70)
    print("\nThis module provides comprehensive comparison of all approaches.")
    print("\nUsage:")
    print("  from results_comparison import compare_all_approaches")
    print("  comparison = compare_all_approaches(results_dict)")
    print("  print_comparison_report(comparison)")
    print("  plot_comparison_charts(comparison, save_dir='./figures')")
