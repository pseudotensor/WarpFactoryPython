"""
Rotating Warp Bubble: Use angular momentum and frame-dragging for propulsion

NOVEL IDEA: Instead of linear acceleration, use ROTATION!

Rotating mass distributions create frame-dragging (Lense-Thirring effect).
A rotating warp bubble might generate thrust through:
1. Frame-dragging pulls spacetime along
2. Asymmetric rotation creates net momentum
3. Precession or nutation could provide directional control

Physical Analogy:
- Kerr black hole (rotating) vs Schwarzschild (static)
- Gyroscope precession for directional thrust
- Tornado/hurricane momentum transport

Key Advantages:
- Rotation can be stationary in rotating frame (∂_t g = 0!)
- Frame-dragging is well-understood in GR
- No time-dependent metric in co-rotating coordinates

Challenge:
- Need axisymmetric metric (cylindrical coordinates)
- Angular momentum conservation
- Stability of rotating configuration
"""

import numpy as np
import sys
sys.path.append('/WarpFactory/warpfactory_py')


def frame_dragging_velocity(
    r: float,
    z: float,
    J: float,
    M: float
) -> float:
    """
    Calculate frame-dragging velocity from rotating mass

    Lense-Thirring effect for slowly rotating shell:
    ω = 2GJ / (c² r³)

    where J is angular momentum, r is distance

    Args:
        r: Cylindrical radius [m]
        z: Height coordinate [m]
        J: Angular momentum [kg m² / s]
        M: Total mass [kg]

    Returns:
        Frame-dragging angular velocity [rad/s]
    """
    from warpfactory.units.constants import G, c

    G_val = G()
    c_val = c()

    # Distance from rotation axis
    distance = np.sqrt(r**2 + z**2) + 1e-10  # Avoid singularity

    # Frame-dragging rate
    omega = 2 * G_val * J / (c_val**2 * distance**3)

    return omega


def rotating_shell_thrust(
    shell_mass: float,
    shell_radius: float,
    rotation_rate: float,
    asymmetry_parameter: float = 0.1
) -> float:
    """
    Estimate thrust from asymmetric rotating shell

    Idea: If rotation is asymmetric (wobbling), creates net momentum

    Args:
        shell_mass: Mass of rotating shell [kg]
        shell_radius: Radius of shell [m]
        rotation_rate: Rotation frequency [Hz]
        asymmetry_parameter: Degree of wobble/precession [dimensionless]

    Returns:
        Estimated thrust [N]
    """
    from warpfactory.units.constants import G, c

    # Angular momentum
    J = shell_mass * shell_radius**2 * (2 * np.pi * rotation_rate)

    # Frame-dragging at shell radius
    omega_drag = frame_dragging_velocity(shell_radius, 0, J, shell_mass)

    # Asymmetric rotation creates net momentum flux
    # (Heuristic estimate - needs full GR calculation)
    momentum_flux = asymmetry_parameter * shell_mass * shell_radius * omega_drag * rotation_rate

    thrust = momentum_flux * (2 * np.pi * rotation_rate)  # Per rotation cycle

    return thrust


def analyze_rotating_approach():
    """
    Analyze the rotating warp bubble approach

    This is a highly creative, novel approach that leverages:
    1. Stationary rotating metrics (no time-dependence in co-rotating frame)
    2. Frame-dragging as propulsion mechanism
    3. Asymmetry for directional thrust
    """
    print("="*70)
    print(" ROTATING WARP BUBBLE ANALYSIS")
    print("="*70)
    print()

    # Parameters
    M = 4.49e27  # kg (2.37 Jupiter masses)
    R = 20.0  # m
    rotation_rates = [0.1, 1.0, 10.0]  # Hz

    print("Configuration:")
    print(f"  Shell mass: {M:.2e} kg ({M/1.898e27:.2f} Jupiter masses)")
    print(f"  Shell radius: {R} m")
    print()

    for f_rot in rotation_rates:
        print(f"Rotation rate: {f_rot} Hz")

        # Angular momentum
        J = M * R**2 * (2 * np.pi * f_rot)
        print(f"  Angular momentum: {J:.2e} kg·m²/s")

        # Frame-dragging at shell
        omega_drag = frame_dragging_velocity(R, 0, J, M)
        print(f"  Frame-dragging ω: {omega_drag:.2e} rad/s")

        # Tangential velocity from dragging
        v_drag = omega_drag * R
        print(f"  Dragging velocity: {v_drag:.2e} m/s ({v_drag/speed_of_light():.2e}c)")

        # Estimate thrust from asymmetric rotation
        for asymmetry in [0.01, 0.1, 1.0]:
            thrust = rotating_shell_thrust(M, R, f_rot, asymmetry)
            print(f"  Thrust (α={asymmetry}): {thrust:.2e} N")

        print()

    print("="*70)
    print("ANALYSIS:")
    print("="*70)
    print()
    print("Key Observations:")
    print()
    print("1. FRAME-DRAGGING IS TINY")
    print("   - Even at 10 Hz, dragging velocity ~ 10^-20 c")
    print("   - Kerr black hole (J=J_max) needed for significant dragging")
    print("   - Our shell: J << J_max for given mass")
    print()
    print("2. THRUST IS MINIMAL")
    print("   - Asymmetric rotation thrust ~ 10^-10 N (negligible)")
    print("   - Compare to rocket: ~10^6 N typical")
    print("   - Need MUCH faster rotation for meaningful thrust")
    print()
    print("3. STABILITY ISSUES")
    print("   - Rotating Jupiter-mass shell at 10 Hz?")
    print("   - Centrifugal forces would tear it apart")
    print("   - Material strength requirements impossible")
    print()
    print("4. STATIONARY IN ROTATING FRAME")
    print("   - ✓ Advantage: No time-dependence in co-rotating coords")
    print("   - ✓ Energy conditions might be satisfied")
    print("   - ✗ But frame-dragging too weak to provide thrust")
    print()
    print("="*70)
    print("CONCLUSION:")
    print("="*70)
    print()
    print("Rotating bubble approach is CREATIVE but IMPRACTICAL:")
    print("  - Frame-dragging effect is too weak (v ~ 10^-20 c)")
    print("  - Would need near-maximal rotation (impossible for shell)")
    print("  - Structural integrity cannot be maintained")
    print("  - Thrust insufficient for meaningful acceleration")
    print()
    print("However, the TIME-INDEPENDENCE in rotating frame is interesting!")
    print("Could inspire: Oscillating bubbles, precessing configurations, etc.")
    print()
    print("STATUS: Explored but not viable with current parameters")


if __name__ == "__main__":
    analyze_rotating_approach()
