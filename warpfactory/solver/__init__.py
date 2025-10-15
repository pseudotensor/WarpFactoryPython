"""
Solver module for Einstein field equations

This module provides numerical solvers for computing:
- Finite difference derivatives
- Christoffel symbols
- Covariant derivatives
- Ricci tensor
- Einstein tensor
- Stress-energy tensor from metrics
"""

from . import finite_differences
from . import christoffel
from . import covariant_derivative
from . import ricci
from . import einstein
from . import energy

__all__ = [
    "finite_differences",
    "christoffel",
    "covariant_derivative",
    "ricci",
    "einstein",
    "energy",
]
