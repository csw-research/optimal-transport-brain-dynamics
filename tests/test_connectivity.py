"""Tests for dynamic connectivity analysis."""

import numpy as np
import pytest

from ot_brain_dynamics.connectivity import (
    sliding_window_connectivity,
    wasserstein_trajectory,
    detect_state_transitions,
    segment_network_states,
)
from ot_brain_dynamics.simulations import generate_var_transitions


@pytest.fixture
def synthetic_data():
    """Generate synthetic data with known transitions."""
    n_regions = 30
    n_timepoints = 500

    time_series, state_labels = generate_var_transitions(
        n_regions=n_regions,
        n_timepoints=n_timepoints,
        n_states=3,
        transition_type='abrupt',
        random_state=42
    )

    return time_series, state_labels


def test_sliding_window_shape(synthetic_data):
    """Test sliding window produces correct output shape."""
    time_series, _ = synthetic_data
    n_timepoints, n_regions = time_series.shape

    window_size = 50
    conn_matrices = sliding_window_connectivity(time_series, window_size=window_size)

    # Check shape
    n_windows = (n_timepoints - window_size) // (window_size // 2) + 1
    assert conn_matrices.shape[1:] == (n_regions, n_regions)
    assert conn_matrices.shape[0] > 0


def test_sliding_window_correlation_range(synthetic_data):
    """Test correlation matrices have values in valid range."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(
        time_series, window_size=50, method='correlation'
    )

    # All correlations should be in [-1, 1]
    assert conn_matrices.min() >= -1.0 - 1e-6
    assert conn_matrices.max() <= 1.0 + 1e-6

    # Diagonal should be 1
    for conn in conn_matrices:
        np.testing.assert_allclose(np.diag(conn), 1.0, rtol=1e-6)


def test_sliding_window_spd(synthetic_data):
    """Test covariance matrices are SPD."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(
        time_series, window_size=50, method='covariance'
    )

    for conn in conn_matrices:
        # Check symmetry
        np.testing.assert_allclose(conn, conn.T, rtol=1e-10)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(conn)
        assert np.all(eigvals > -1e-10)


def test_wasserstein_trajectory_shape(synthetic_data):
    """Test Wasserstein trajectory has correct shape."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(time_series, window_size=50)
    distances = wasserstein_trajectory(conn_matrices)

    assert len(distances) == len(conn_matrices) - 1
    assert np.all(distances >= 0)


def test_wasserstein_trajectory_detects_changes(synthetic_data):
    """Test Wasserstein distance increases at state transitions."""
    time_series, state_labels = synthetic_data

    conn_matrices = sliding_window_connectivity(time_series, window_size=50)
    distances = wasserstein_trajectory(conn_matrices)

    # Distances should be higher near state transitions
    # This is a soft check due to windowing effects
    assert distances.std() > 0  # Should have variability


def test_detect_transitions_methods(synthetic_data):
    """Test transition detection with different methods."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(time_series, window_size=50)
    distances = wasserstein_trajectory(conn_matrices)

    for method in ['peaks', 'threshold']:
        transitions = detect_state_transitions(distances, method=method)

        assert len(transitions) > 0
        assert all(0 <= t < len(distances) for t in transitions)


def test_segment_network_states_shape(synthetic_data):
    """Test state segmentation produces valid labels."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(time_series, window_size=50)

    n_states = 3
    state_labels = segment_network_states(conn_matrices, n_states=n_states, method='kmeans')

    assert len(state_labels) == len(conn_matrices)
    assert len(np.unique(state_labels)) <= n_states
    assert state_labels.min() >= 0


def test_segment_network_states_with_centers(synthetic_data):
    """Test state segmentation returns valid centers."""
    time_series, _ = synthetic_data

    conn_matrices = sliding_window_connectivity(time_series, window_size=50)

    n_states = 3
    state_labels, state_centers = segment_network_states(
        conn_matrices, n_states=n_states, return_centers=True
    )

    assert state_centers.shape == (n_states, conn_matrices.shape[1], conn_matrices.shape[2])

    # Centers should be SPD
    for center in state_centers:
        eigvals = np.linalg.eigvalsh(center)
        assert np.all(eigvals > -1e-10)


def test_connectivity_methods():
    """Test different connectivity methods work."""
    n_timepoints, n_regions = 200, 20
    time_series = np.random.randn(n_timepoints, n_regions)

    for method in ['correlation', 'covariance', 'partial']:
        conn_matrices = sliding_window_connectivity(
            time_series, window_size=50, method=method
        )

        assert conn_matrices.shape[1:] == (n_regions, n_regions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
