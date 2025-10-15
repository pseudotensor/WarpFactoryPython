"""
WarpFactory Test Suite

This package contains comprehensive unit tests for the WarpFactory Python package.

Test Modules:
    test_units: Tests for physical constants and unit conversions
    test_core: Tests for Tensor class and tensor operations
    test_metrics: Tests for all metric implementations (Minkowski, Alcubierre, etc.)
    test_solver: Tests for finite differences, Ricci tensor, Einstein tensor
    test_analyzer: Tests for energy conditions, scalars, and analyzer utilities
    test_visualizer: Tests for plotting and visualization utilities

Running Tests:
    # Run all tests
    pytest warpfactory/tests/

    # Run specific test module
    pytest warpfactory/tests/test_units.py

    # Run with verbose output
    pytest -v warpfactory/tests/

    # Run with coverage
    pytest --cov=warpfactory warpfactory/tests/

Fixtures:
    Common test fixtures are defined in conftest.py and include:
    - Grid sizes (small, medium)
    - Metric tensors (Minkowski on various grids)
    - Test parameters (tolerances, world centers, etc.)
"""

__version__ = "1.0.0"
