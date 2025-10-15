"""
Van Den Broeck warp drive metric

The Van Den Broeck metric is a modification of the Alcubierre metric that
reduces the total energy requirements by using a spatial expansion factor
to create a larger interior volume while maintaining a small exterior profile.
"""

from .van_den_broeck import get_van_den_broeck_metric

__all__ = ["get_van_den_broeck_metric"]
