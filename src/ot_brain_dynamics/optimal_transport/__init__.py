"""
Optimal transport algorithms for symmetric positive definite matrices.

This module implements Wasserstein distances, geodesics, barycenters, and
transport plans for SPD matrices representing brain connectivity states.
"""

from ot_brain_dynamics.optimal_transport.spd_geometry import (
    sqrtm_spd,
    logm_spd,
    expm_spd,
    geodesic_spd,
    distance_riemannian,
)
from ot_brain_dynamics.optimal_transport.wasserstein import (
    wasserstein_spd,
    wasserstein_barycenter_spd,
    optimal_transport_plan,
)
from ot_brain_dynamics.optimal_transport.sinkhorn import (
    sinkhorn_distance,
    sinkhorn_barycenter,
)

__all__ = [
    "sqrtm_spd",
    "logm_spd",
    "expm_spd",
    "geodesic_spd",
    "distance_riemannian",
    "wasserstein_spd",
    "wasserstein_barycenter_spd",
    "optimal_transport_plan",
    "sinkhorn_distance",
    "sinkhorn_barycenter",
]
