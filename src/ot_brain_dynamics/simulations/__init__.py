"""
Simulation framework for generating synthetic fMRI time series.

Provides functions to generate realistic time series data with known
ground truth network transitions for validation and benchmarking.
"""

from ot_brain_dynamics.simulations.var_models import (
    generate_var_time_series,
    generate_var_transitions,
    TimeVaryingVAR,
)
from ot_brain_dynamics.simulations.network_states import (
    generate_network_states,
    create_modular_network,
    create_hub_network,
    create_random_network,
)
from ot_brain_dynamics.simulations.noise_models import (
    add_physiological_noise,
    add_motion_artifacts,
    add_scanner_noise,
)

__all__ = [
    "generate_var_time_series",
    "generate_var_transitions",
    "TimeVaryingVAR",
    "generate_network_states",
    "create_modular_network",
    "create_hub_network",
    "create_random_network",
    "add_physiological_noise",
    "add_motion_artifacts",
    "add_scanner_noise",
]
