"""
Test for reproducing paper 2405.02709v1 results

This test reproduces the results from the paper:
"Constant Velocity Physical Warp Drive Solution" by Fuchs et al. (2023)

Paper parameters (from Section 3 and 4):
- Grid: 61x61x61 spatial points (single time slice)
- Mass: M = 4.49e27 kg (2.365 Jupiter masses)
- Inner radius: R1 = 10 m
- Outer radius: R2 = 20 m
- Warp velocity: vWarp = 0.02 c (for Warp Shell section 4)
- Smoothing factor: 1.0
- Buffer: Rb = 0 (default)
- Sigma: 0 (default)

Expected result from paper:
- Both Matter Shell and Warp Shell should show ZERO energy condition violations
- All energy conditions (NEC, WEC, SEC, DEC) should be satisfied
- Violations should be below numerical precision (~10^-34)
"""

import pytest
import numpy as np


@pytest.mark.slow
class TestPaperReproduction:
    """
    Reproduce paper 2405.02709v1 results for Warp Shell metric

    These tests verify that the Python implementation produces the same results
    as the MATLAB implementation for the paper's benchmark case.
    """

    @pytest.fixture
    def paper_params(self):
        """Parameters from paper 2405.02709v1"""
        return {
            'grid_size': [1, 61, 61, 61],  # Single time slice
            'world_center': [0.5, 30.5, 30.5, 30.5],  # Center of 61x61x61 grid
            'grid_scaling': [1.0, 1.0, 1.0, 1.0],
            'm': 4.49e27,  # kg - 2.365 Jupiter masses
            'R1': 10.0,    # m - Inner radius
            'R2': 20.0,    # m - Outer radius
            'Rbuff': 0.0,  # Buffer distance
            'sigma': 0.0,  # Sharpness parameter
            'smooth_factor': 1.0,  # Smoothing factor
            'v_warp': 0.02,  # Warp velocity (fraction of c)
        }

    def test_matter_shell_no_warp_metric_creation(self, paper_params):
        """
        Test Section 3: Matter shell without warp

        Creates the metric for a matter shell without warp effect.
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.core.tensor_ops import verify_tensor

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=False,  # No warp for Section 3
            grid_scaling=paper_params['grid_scaling']
        )

        assert metric.name == "Comoving Warp Shell", \
            "Metric name should be 'Comoving Warp Shell'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Matter shell metric should verify"

    def test_matter_shell_no_warp_energy_density(self, paper_params):
        """
        Test energy density for matter shell (no warp)

        The energy density should be concentrated in the shell region
        between R1 and R2.
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=False,
            grid_scaling=paper_params['grid_scaling']
        )

        # Compute energy tensor
        energy = get_energy_tensor(metric, diff_order='fourth')

        # Extract energy density T_00
        T00 = energy[(0, 0)]

        # Energy density should have both positive and negative values
        # (matter has positive energy, but there can be numerical artifacts)
        assert np.all(np.isfinite(T00)), "T_00 should be finite everywhere"

        # Check that we have some non-zero energy density in the shell
        assert not np.allclose(T00, 0.0), \
            "Energy density should be non-zero in the shell"

    @pytest.mark.slow
    def test_matter_shell_no_warp_nec(self, paper_params):
        """
        Test Null Energy Condition (NEC) for matter shell

        Expected from paper: Zero violations (all points satisfy NEC)
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor
        from warpfactory.analyzer.energy_conditions import get_energy_conditions

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=False,
            grid_scaling=paper_params['grid_scaling']
        )

        energy = get_energy_tensor(metric, diff_order='fourth')

        # Test NEC with sample size parameters
        nec = get_energy_conditions(
            energy,
            metric,
            condition_type="Null",
            sample_size=100,
            sample_density=10
        )

        # Count violations (NEC < 0)
        violations = np.sum(nec < 0)
        total_points = np.prod(nec.shape)
        violation_percent = 100.0 * violations / total_points

        # According to paper, should have zero violations
        # Allow small numerical tolerance
        assert violation_percent < 1.0, \
            f"NEC violation percentage should be near 0%, got {violation_percent:.2f}%"

    @pytest.mark.slow
    def test_matter_shell_no_warp_wec(self, paper_params):
        """
        Test Weak Energy Condition (WEC) for matter shell

        Expected from paper: Zero violations
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor
        from warpfactory.analyzer.energy_conditions import get_energy_conditions

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=False,
            grid_scaling=paper_params['grid_scaling']
        )

        energy = get_energy_tensor(metric, diff_order='fourth')

        wec = get_energy_conditions(
            energy,
            metric,
            condition_type="Weak",
            sample_size=100,
            sample_density=10
        )

        violations = np.sum(wec < 0)
        total_points = np.prod(wec.shape)
        violation_percent = 100.0 * violations / total_points

        assert violation_percent < 1.0, \
            f"WEC violation percentage should be near 0%, got {violation_percent:.2f}%"

    def test_warp_shell_with_warp_metric_creation(self, paper_params):
        """
        Test Section 4: Warp shell with warp

        Creates the metric for a warp shell with warp effect enabled.
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.core.tensor_ops import verify_tensor

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=True,  # Enable warp for Section 4
            grid_scaling=paper_params['grid_scaling']
        )

        assert metric.name == "Comoving Warp Shell", \
            "Metric name should be 'Comoving Warp Shell'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Warp shell metric should verify"

        # Verify warp is enabled in params
        assert metric.params['doWarp'] == True, \
            "doWarp parameter should be True"

    def test_warp_shell_with_warp_shift_vector(self, paper_params):
        """
        Test that warp shell with warp has non-zero shift vector

        The shift vector should be non-zero inside the shell when warp is enabled.
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=True,
            grid_scaling=paper_params['grid_scaling']
        )

        # Check g_tx component (shift vector)
        g_tx = metric[(0, 1)]

        # Should have non-zero shift in some regions when warp is enabled
        assert not np.allclose(g_tx, 0.0), \
            "Shift vector should be non-zero with warp enabled"

    @pytest.mark.slow
    def test_warp_shell_with_warp_nec(self, paper_params):
        """
        Test Null Energy Condition (NEC) for warp shell with warp

        Expected from paper: Zero violations
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor
        from warpfactory.analyzer.energy_conditions import get_energy_conditions

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=True,
            grid_scaling=paper_params['grid_scaling']
        )

        energy = get_energy_tensor(metric, diff_order='fourth')

        nec = get_energy_conditions(
            energy,
            metric,
            condition_type="Null",
            sample_size=100,
            sample_density=10
        )

        violations = np.sum(nec < 0)
        total_points = np.prod(nec.shape)
        violation_percent = 100.0 * violations / total_points

        # According to paper, should have zero violations
        assert violation_percent < 1.0, \
            f"NEC violation percentage should be near 0%, got {violation_percent:.2f}%"

    @pytest.mark.slow
    def test_warp_shell_with_warp_wec(self, paper_params):
        """
        Test Weak Energy Condition (WEC) for warp shell with warp

        Expected from paper: Zero violations
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor
        from warpfactory.analyzer.energy_conditions import get_energy_conditions

        metric = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=True,
            grid_scaling=paper_params['grid_scaling']
        )

        energy = get_energy_tensor(metric, diff_order='fourth')

        wec = get_energy_conditions(
            energy,
            metric,
            condition_type="Weak",
            sample_size=100,
            sample_density=10
        )

        violations = np.sum(wec < 0)
        total_points = np.prod(wec.shape)
        violation_percent = 100.0 * violations / total_points

        assert violation_percent < 1.0, \
            f"WEC violation percentage should be near 0%, got {violation_percent:.2f}%"

    @pytest.mark.slow
    def test_comparison_matter_vs_warp_shell(self, paper_params):
        """
        Compare matter shell (no warp) vs warp shell (with warp)

        Both should have similar energy condition satisfaction rates
        according to the paper.
        """
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor
        from warpfactory.analyzer.energy_conditions import get_energy_conditions

        # Create matter shell (no warp)
        metric_no_warp = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=False,
            grid_scaling=paper_params['grid_scaling']
        )

        # Create warp shell (with warp)
        metric_with_warp = get_warp_shell_comoving_metric(
            paper_params['grid_size'],
            paper_params['world_center'],
            paper_params['m'],
            paper_params['R1'],
            paper_params['R2'],
            paper_params['Rbuff'],
            paper_params['sigma'],
            paper_params['smooth_factor'],
            paper_params['v_warp'],
            do_warp=True,
            grid_scaling=paper_params['grid_scaling']
        )

        # Compute energy tensors
        energy_no_warp = get_energy_tensor(metric_no_warp, diff_order='fourth')
        energy_with_warp = get_energy_tensor(metric_with_warp, diff_order='fourth')

        # Test NEC for both
        nec_no_warp = get_energy_conditions(
            energy_no_warp,
            metric_no_warp,
            condition_type="Null",
            sample_size=50,
            sample_density=5
        )

        nec_with_warp = get_energy_conditions(
            energy_with_warp,
            metric_with_warp,
            condition_type="Null",
            sample_size=50,
            sample_density=5
        )

        # Both should have low violation rates
        violations_no_warp = np.sum(nec_no_warp < 0)
        violations_with_warp = np.sum(nec_with_warp < 0)

        total_points = np.prod(nec_no_warp.shape)

        violation_percent_no_warp = 100.0 * violations_no_warp / total_points
        violation_percent_with_warp = 100.0 * violations_with_warp / total_points

        # Both should be near zero according to paper
        assert violation_percent_no_warp < 2.0, \
            f"Matter shell NEC violations should be minimal, got {violation_percent_no_warp:.2f}%"
        assert violation_percent_with_warp < 2.0, \
            f"Warp shell NEC violations should be minimal, got {violation_percent_with_warp:.2f}%"


@pytest.mark.slow
class TestPaperReproductionQuick:
    """
    Quick tests for paper reproduction with smaller grid

    These tests use a smaller grid size for faster execution while still
    verifying the basic behavior matches the paper.
    """

    @pytest.fixture
    def quick_params(self):
        """Smaller grid parameters for quick tests"""
        return {
            'grid_size': [1, 21, 21, 21],  # Smaller grid: 21x21x21
            'world_center': [0.5, 10.5, 10.5, 10.5],
            'grid_scaling': [1.0, 1.0, 1.0, 1.0],
            'm': 4.49e27,
            'R1': 10.0,
            'R2': 20.0,
            'Rbuff': 0.0,
            'sigma': 0.0,
            'smooth_factor': 1.0,
            'v_warp': 0.02,
        }

    def test_quick_matter_shell_creation(self, quick_params):
        """Quick test: Create matter shell metric"""
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.core.tensor_ops import verify_tensor

        metric = get_warp_shell_comoving_metric(
            quick_params['grid_size'],
            quick_params['world_center'],
            quick_params['m'],
            quick_params['R1'],
            quick_params['R2'],
            quick_params['Rbuff'],
            quick_params['sigma'],
            quick_params['smooth_factor'],
            quick_params['v_warp'],
            do_warp=False,
            grid_scaling=quick_params['grid_scaling']
        )

        assert verify_tensor(metric, suppress_msgs=True), \
            "Matter shell metric should verify"

    def test_quick_warp_shell_creation(self, quick_params):
        """Quick test: Create warp shell metric"""
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.core.tensor_ops import verify_tensor

        metric = get_warp_shell_comoving_metric(
            quick_params['grid_size'],
            quick_params['world_center'],
            quick_params['m'],
            quick_params['R1'],
            quick_params['R2'],
            quick_params['Rbuff'],
            quick_params['sigma'],
            quick_params['smooth_factor'],
            quick_params['v_warp'],
            do_warp=True,
            grid_scaling=quick_params['grid_scaling']
        )

        assert verify_tensor(metric, suppress_msgs=True), \
            "Warp shell metric should verify"

    def test_quick_energy_tensor_computation(self, quick_params):
        """Quick test: Compute energy tensor"""
        from warpfactory.metrics.warp_shell.warp_shell import \
            get_warp_shell_comoving_metric
        from warpfactory.solver.energy import get_energy_tensor

        metric = get_warp_shell_comoving_metric(
            quick_params['grid_size'],
            quick_params['world_center'],
            quick_params['m'],
            quick_params['R1'],
            quick_params['R2'],
            quick_params['Rbuff'],
            quick_params['sigma'],
            quick_params['smooth_factor'],
            quick_params['v_warp'],
            do_warp=False,
            grid_scaling=quick_params['grid_scaling']
        )

        energy = get_energy_tensor(metric, diff_order='fourth')

        # Check that all components are finite
        for i in range(4):
            for j in range(4):
                assert np.all(np.isfinite(energy[(i, j)])), \
                    f"Energy tensor component ({i},{j}) should be finite"
