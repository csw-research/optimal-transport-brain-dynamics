"""
Dynamic functional connectivity estimation and analysis.

Provides methods for computing time-varying connectivity using sliding windows,
robust covariance estimation, and Wasserstein-based tracking of network evolution.
"""

from ot_brain_dynamics.connectivity.sliding_window import (
    sliding_window_connectivity,
    adaptive_window_connectivity,
    estimate_optimal_window_size,
)
from ot_brain_dynamics.connectivity.covariance_estimation import (
    robust_covariance_estimate,
    shrinkage_covariance,
    graphical_lasso_covariance,
)
from ot_brain_dynamics.connectivity.dynamic_analysis import (
    wasserstein_trajectory,
    detect_state_transitions,
    segment_network_states,
    compute_state_statistics,
)

__all__ = [
    "sliding_window_connectivity",
    "adaptive_window_connectivity",
    "estimate_optimal_window_size",
    "robust_covariance_estimate",
    "shrinkage_covariance",
    "graphical_lasso_covariance",
    "wasserstein_trajectory",
    "detect_state_transitions",
    "segment_network_states",
    "compute_state_statistics",
]
