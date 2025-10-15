"""
Alcubierre warp drive metric

The Alcubierre metric is a solution to Einstein's field equations that
allows for faster-than-light travel by contracting space in front of
a spacecraft and expanding space behind it.
"""

from .alcubierre import get_alcubierre_metric, shape_function_alcubierre

__all__ = ["get_alcubierre_metric", "shape_function_alcubierre"]
