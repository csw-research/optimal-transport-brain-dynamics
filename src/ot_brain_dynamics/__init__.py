"""
Optimal Transport for Dynamic Brain Connectivity Analysis.

A mathematical framework for analyzing time-varying functional brain connectivity
using optimal transport theory on the manifold of correlation/covariance matrices.
"""

__version__ = "0.1.0"
__author__ = "CSW Research"

from ot_brain_dynamics import optimal_transport, connectivity, simulations, visualization

__all__ = ["optimal_transport", "connectivity", "simulations", "visualization"]
