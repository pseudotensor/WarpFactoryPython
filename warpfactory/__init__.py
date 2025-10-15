"""
WarpFactory: Numerical toolkit for analyzing warp drive spacetimes

This package provides tools for analyzing warp drive spacetimes using
Einstein's theory of General Relativity.
"""

__version__ = "1.0.0"
__author__ = "Christopher Helmerich, Jared Fuchs, and Contributors"

# Import submodules for convenience
from . import units
from . import metrics
from . import solver
from . import analyzer
from . import visualizer

__all__ = [
    "units",
    "metrics",
    "solver",
    "analyzer",
    "visualizer",
]
