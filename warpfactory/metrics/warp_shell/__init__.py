"""
Warp Shell metric module

Implements the comoving warp shell metric as described in:
https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa
"""

from .warp_shell import get_warp_shell_comoving_metric

__all__ = ["get_warp_shell_comoving_metric"]
