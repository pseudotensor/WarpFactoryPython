#!/usr/bin/env python3
"""
Full Adiabatic Test - Actually compute energy conditions

This will take several minutes but will give REAL results.
"""

import numpy as np
import sys
import pickle
import time as time_module
sys.path.append('/WarpFactory/warpfactory_py')

from adiabatic_simulation import simulate_adiabatic_acceleration, analyze_scaling_law


if __name__ == "__main__":
    print("FULL ADIABATIC ACCELERATION TEST")
    print("="*80)
    print()
    print("This will test if adiabatic evolution achieves zero violations")
    print("Testing at T=100s with moderate resolution")
    print()
    print("Expected runtime: ~3-5 minutes")
    print()

    start_time = time_module.time()

    # Run with energy condition computation
    results = simulate_adiabatic_acceleration(
        v_final=0.02,
        T_accel=100.0,
        num_time_steps=6,  # Sample fewer points for speed
        grid_size=[1, 15, 15, 15],  # Moderate resolution
        compute_full_conditions=True
    )

    elapsed = time_module.time() - start_time

    print()
    print("="*80)
    print(" RESULTS")
    print("="*80)
    print()
    print(f"Total runtime: {elapsed:.1f} seconds")
    print()

    analyze_scaling_law(results)

    # Save results
    output_file = 'acceleration_research3/adiabatic_results_T100.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_file}")
    print()

    # Summary
    null_violations = [v for v in results['violations']['null'] if v is not None]

    if len(null_violations) > 0:
        worst_violation = min(null_violations)
        print("="*80)
        print(" BREAKTHROUGH ASSESSMENT")
        print("="*80)
        print()
        print(f"Worst Null Energy Condition violation: {worst_violation:.2e}")
        print()

        # Compare with predictions
        predicted = (0.02/100)**2
        print(f"Predicted (from (dv/dt)²): {predicted:.2e}")
        print()

        # Compare with multi-shell
        multi_shell_best = 2.61e85
        if abs(worst_violation) < multi_shell_best:
            improvement = multi_shell_best / abs(worst_violation)
            print(f"Improvement vs multi-shell: {improvement:.2e}x")
        print()

        # Check if effectively zero
        if abs(worst_violation) < 1e-15:
            print("✓✓✓ EFFECTIVELY ZERO (below numerical precision)!")
            print("    BREAKTHROUGH ACHIEVED!")
        elif abs(worst_violation) < 1e-10:
            print("✓✓ NEGLIGIBLE (below physical significance)")
            print("   Near-zero solution found!")
        elif abs(worst_violation) < 1e-5:
            print("✓ VERY SMALL (orders better than classical)")
            print("  Significant progress")
        else:
            print("  Still has violations, but may scale down with longer T")
            print(f"  Try T=1000s or T=10000s for better results")

        print()
    else:
        print("No energy condition results (computation not run)")

    print("Next steps:")
    print("  1. Run at longer times: T=1000s, T=10000s")
    print("  2. Verify scaling law")
    print("  3. Compare with multi-shell quantitatively")
    print("  4. If violations < 10^-10, this is THE solution!")
