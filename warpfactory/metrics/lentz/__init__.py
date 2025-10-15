"""
Lentz warp drive metric

The Lentz metric is a soliton-based warp drive solution that uses a
discontinuous shift vector template to create a warp bubble geometry.
Unlike the Alcubierre metric which uses smooth shape functions, the Lentz
metric uses a piecewise constant shift vector defined over specific regions
in space.
"""

from .lentz import get_lentz_metric, get_warp_factor_by_region

__all__ = ["get_lentz_metric", "get_warp_factor_by_region"]
