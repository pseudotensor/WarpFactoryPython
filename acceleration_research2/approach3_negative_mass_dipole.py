"""
Negative Mass Dipole: Use positive/negative mass pair for propulsion

REVOLUTIONARY IDEA: Create a mass dipole configuration!

Place negative mass BEHIND the bubble and positive mass IN FRONT.
This creates:
1. Gravitational repulsion from negative mass (pushes forward)
2. Gravitational attraction to positive mass (pulls forward)
3. Net thrust without violating momentum conservation
4. The "runaway motion" paradox becomes a FEATURE, not a bug!

Physical Basis:
- Negative mass solutions exist in GR (exotic matter)
- Bondi (1957): Negative mass repels everything, including positive mass
- Forward (1990): Negative mass behind, positive ahead = constant acceleration!
- Runaway paradox: Positive mass attracted to negative, negative repelled from positive
  → Both accelerate in same direction indefinitely!

Key Insight:
- This is NOT the same as time-varying single mass
- It's a CONFIGURATION that's static in its own frame
- Acceleration comes from mass distribution, not time-dependence
- Could be stationary metric in non-inertial coordinates!

Challenge:
- Negative mass is exotic (though less exotic than warp drive itself!)
- Stability of configuration
- How to create/maintain negative mass regions
"""

import numpy as np
import sys
sys.path.append('/WarpFactory/warpfactory_py')

from warpfactory.units.constants import G, c as speed_of_light


def gravitational_force_dipole(
    m_positive: float,
    m_negative: float,
    separation: float,
    test_mass: float = 1.0
) -> float:
    """
    Calculate gravitational force on test mass between positive/negative masses

    Configuration:
        [Negative Mass] <--- separation ---> [Test Mass] <--- d ---> [Positive Mass]

    Args:
        m_positive: Positive mass value [kg]
        m_negative: Negative mass value [kg] (enter as positive number!)
        separation: Distance between masses [m]
        test_mass: Mass of object to be accelerated [kg]

    Returns:
        Net force [N] (positive = rightward toward positive mass)
    """
    G_val = G()

    # Position test mass at origin
    # Negative mass at x = -separation/2
    # Positive mass at x = +separation/2

    # Force from negative mass (repulsive, points away)
    F_negative = -G_val * (-m_negative) * test_mass / (separation/2)**2  # Note negative mass!
    # Negative mass creates repulsive force, pushing test mass rightward

    # Force from positive mass (attractive, points toward)
    F_positive = G_val * m_positive * test_mass / (separation/2)**2
    # Positive mass creates attractive force, pulling test mass rightward

    # Both point in SAME direction!
    F_net = F_negative + F_positive

    return F_net


def runaway_acceleration_analysis():
    """
    Analyze the famous "runaway motion" paradox

    Bondi (1957) showed:
    - Negative mass repels positive mass
    - Positive mass attracts negative mass
    - Result: Both accelerate in same direction!
    - This seems paradoxical but is consistent with F=ma for negative mass

    For our purpose: This is EXACTLY what we want for propulsion!
    """
    print("="*70)
    print(" NEGATIVE MASS DIPOLE PROPULSION")
    print("="*70)
    print()

    # Configuration
    m_pos = 2.37e27  # kg (Jupiter-mass)
    m_neg = 2.37e27  # kg (same magnitude, opposite sign)
    separation = 50.0  # m
    test_mass = 100000.0  # kg (spacecraft)

    print("Configuration:")
    print(f"  Positive mass: +{m_pos:.2e} kg (ahead of spacecraft)")
    print(f"  Negative mass: -{m_neg:.2e} kg (behind spacecraft)")
    print(f"  Separation: {separation} m")
    print(f"  Spacecraft mass: {test_mass:.2e} kg")
    print()

    # Calculate forces
    F_net = gravitational_force_dipole(m_pos, m_neg, separation, test_mass)

    # Acceleration
    a = F_net / test_mass

    print("Forces on spacecraft:")
    print(f"  From negative mass (repulsion): {F_net/2:.2e} N →")
    print(f"  From positive mass (attraction): {F_net/2:.2e} N →")
    print(f"  Net force: {F_net:.2e} N")
    print(f"  Acceleration: {a:.2e} m/s²")
    print(f"  Acceleration: {a/9.81:.2e} g")
    print()

    # Time to reach 0.02c
    c_val = speed_of_light()
    v_target = 0.02 * c_val
    t_accel = v_target / a

    print(f"Time to reach 0.02c: {t_accel:.2e} s ({t_accel/86400:.1f} days)")
    print()

    print("="*70)
    print("RUNAWAY MOTION PARADOX:")
    print("="*70)
    print()
    print("What happens to the masses themselves?")
    print()
    print("Positive mass (+M):")
    print("  Force from negative: F = -G(-M)(+M)/d² = +G M²/d² (repulsion)")
    print("  Acceleration: a = +G M/d² (accelerates FORWARD)")
    print()
    print("Negative mass (-M):")
    print("  Force from positive: F = G(+M)(-M)/d² = -G M²/d² (attraction)")
    print("  But F=ma for negative mass: a = F/(-M) = +G M/d² (also FORWARD!)")
    print()
    print("Result: BOTH masses accelerate forward together!")
    print("        The separation remains constant")
    print("        Perpetual acceleration with NO energy input!")
    print()
    print("This violates energy conservation UNLESS:")
    print("  - Negative mass has negative energy: E = mc² → E = -|m|c²")
    print("  - Total energy: E_pos + E_neg = (+M - M)c² = 0")
    print("  - Energy is conserved as system accelerates!")
    print()

    print("="*70)
    print("PRACTICAL CONSIDERATIONS:")
    print("="*70)
    print()
    print("Advantages:")
    print("  ✓ Constant acceleration without energy input")
    print("  ✓ Metric could be static in accelerating frame")
    print("  ✓ No time-dependence problem")
    print("  ✓ Energy conditions might be satisfiable")
    print()
    print("Challenges:")
    print("  ✗ Requires negative mass (highly exotic!)")
    print("  ✗ Negative mass never observed in nature")
    print("  ✗ Quantum field theory generally forbids negative mass")
    print("  ✗ Stability: Configuration might be unstable")
    print("  ✗ Creation: How to generate negative mass?")
    print()
    print("Physics Status:")
    print("  - Consistent with GR equations (solutions exist)")
    print("  - But violates energy conditions generically")
    print("  - Quantum corrections unknown")
    print("  - Considered 'exotic' even by warp drive standards")
    print()

    print("="*70)
    print("CREATIVE VARIATIONS:")
    print("="*70)
    print()
    print("1. Casimir Negative Energy:")
    print("   - Casimir effect creates ρ < 0 between plates")
    print("   - Could this substitute for negative mass?")
    print("   - Magnitude tiny (~10^-7 J/m³), need amplification")
    print()
    print("2. Negative Mass 'Wake':")
    print("   - Place negative mass behind spacecraft")
    print("   - Continuous generation as you move?")
    print("   - Like creating contrail")
    print()
    print("3. Oscillating Positive/Negative:")
    print("   - Rapidly alternate positive and negative regions")
    print("   - Time-averaged might give net thrust")
    print("   - Related to dynamical Casimir effect")
    print()

    print("="*70)
    print("CONCLUSION:")
    print("="*70)
    print()
    print("STATUS: Extremely creative but requires exotic physics")
    print()
    print("The negative mass dipole SOLVES the time-dependence problem")
    print("by using a static configuration that naturally accelerates.")
    print()
    print("However, it trades one exotic requirement (time-dependent metric)")
    print("for another (negative mass). May not be an improvement overall.")
    print()
    print("INTERESTING DIRECTION: Casimir energy as negative mass substitute")
    print("  - More physically grounded")
    print("  - Quantum vacuum engineering")
    print("  - Worth exploring in QFT framework")


if __name__ == "__main__":
    runaway_acceleration_analysis()
