"""Tests for simulation framework."""

import numpy as np
import pytest

from ot_brain_dynamics.simulations import (
    generate_var_transitions,
    generate_var_time_series,
    TimeVaryingVAR,
    generate_network_states,
    create_modular_network,
    create_hub_network,
)


def test_generate_var_transitions_shape():
    """Test generated time series has correct shape."""
    n_regions = 50
    n_timepoints = 500
    n_states = 3

    time_series, state_labels = generate_var_transitions(
        n_regions=n_regions,
        n_timepoints=n_timepoints,
        n_states=n_states,
        transition_type='smooth',
        random_state=42
    )

    assert time_series.shape == (n_timepoints, n_regions)
    assert state_labels.shape == (n_timepoints,)
    assert len(np.unique(state_labels)) <= n_states


def test_generate_var_transitions_types():
    """Test different transition types work."""
    n_regions = 30
    n_timepoints = 300

    for trans_type in ['smooth', 'abrupt', 'oscillating']:
        time_series, state_labels = generate_var_transitions(
            n_regions=n_regions,
            n_timepoints=n_timepoints,
            transition_type=trans_type,
            random_state=42
        )

        assert time_series.shape == (n_timepoints, n_regions)
        assert np.isfinite(time_series).all()


def test_generate_var_time_series_stationarity():
    """Test stationary VAR generates reasonable data."""
    n_regions = 20
    n_timepoints = 200

    # Simple connectivity matrix
    A = np.random.randn(n_regions, n_regions) * 0.1
    A = A * 0.5 / np.max(np.abs(np.linalg.eigvals(A)))  # Ensure stability

    time_series = generate_var_time_series(
        n_regions=n_regions,
        n_timepoints=n_timepoints,
        connectivity_matrix=A,
        random_state=42
    )

    assert time_series.shape == (n_timepoints, n_regions)
    assert np.isfinite(time_series).all()

    # Check reasonable variance (not exploding)
    assert time_series.std() < 100


def test_time_varying_var_model():
    """Test TimeVaryingVAR class."""
    n_regions = 30
    n_timepoints = 400

    model = TimeVaryingVAR(n_regions=n_regions)
    time_series, state_labels = model.generate(
        n_timepoints=n_timepoints,
        n_states=3,
        transition_type='smooth',
        random_state=42
    )

    assert time_series.shape == (n_timepoints, n_regions)
    assert len(model.A_matrices_) == n_timepoints


def test_network_states_generation():
    """Test generation of network state matrices."""
    n_states = 4
    n_regions = 40

    states = generate_network_states(
        n_states=n_states,
        n_regions=n_regions,
        sparsity=0.2,
        spectral_radius=0.85,
        random_state=42
    )

    assert len(states) == n_states

    for state in states:
        assert state.shape == (n_regions, n_regions)

        # Check spectral radius constraint
        eigvals = np.linalg.eigvals(state)
        max_eigval = np.max(np.abs(eigvals))
        assert max_eigval <= 0.85 + 1e-6


def test_modular_network_structure():
    """Test modular network has block structure."""
    n_regions = 50
    n_modules = 5

    A = create_modular_network(n_regions, n_modules=n_modules, sparsity=0.2)

    assert A.shape == (n_regions, n_regions)

    # Check sparsity roughly correct
    nonzero_ratio = (A != 0).sum() / (n_regions ** 2)
    assert 0.05 < nonzero_ratio < 0.5  # Rough check


def test_hub_network_structure():
    """Test hub network has dense connections to hubs."""
    n_regions = 50
    n_hubs = 5

    A = create_hub_network(n_regions, n_hubs=n_hubs, sparsity=0.2)

    assert A.shape == (n_regions, n_regions)

    # Hub nodes (first n_hubs) should have more connections
    hub_degree = (A[:n_hubs, :] != 0).sum(axis=1).mean()
    periphery_degree = (A[n_hubs:, :] != 0).sum(axis=1).mean()

    assert hub_degree > periphery_degree


def test_reproducibility():
    """Test random seed ensures reproducibility."""
    n_regions = 30
    n_timepoints = 200

    ts1, states1 = generate_var_transitions(
        n_regions, n_timepoints, random_state=42
    )
    ts2, states2 = generate_var_transitions(
        n_regions, n_timepoints, random_state=42
    )

    np.testing.assert_array_equal(ts1, ts2)
    np.testing.assert_array_equal(states1, states2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
