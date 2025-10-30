"""
Visualization tools for dynamic connectivity and optimal transport analysis.

Provides plotting functions for connectivity matrices, geodesics, transport plans,
and network state dynamics.
"""

from ot_brain_dynamics.visualization.connectivity_plots import (
    plot_connectivity_matrix,
    plot_connectivity_evolution,
    plot_state_timeline,
)
from ot_brain_dynamics.visualization.geodesic_plots import (
    plot_geodesic_path,
    plot_wasserstein_trajectory,
    plot_distance_matrix,
)
from ot_brain_dynamics.visualization.transport_plots import (
    plot_transport_plan,
    plot_network_reorganization,
)

__all__ = [
    "plot_connectivity_matrix",
    "plot_connectivity_evolution",
    "plot_state_timeline",
    "plot_geodesic_path",
    "plot_wasserstein_trajectory",
    "plot_distance_matrix",
    "plot_transport_plan",
    "plot_network_reorganization",
]
