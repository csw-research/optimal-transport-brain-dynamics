"""Tests for Wasserstein distance and optimal transport."""

import numpy as np
import pytest

from ot_brain_dynamics.optimal_transport.wasserstein import (
    wasserstein_spd,
    wasserstein_barycenter_spd,
    wasserstein_distance_matrix,
    wasserstein_trajectory_distance,
)


@pytest.fixture
def correlation_matrix():
    """Create a valid correlation matrix."""
    n = 20
    # Create from random covariance
    A = np.random.randn(n, n) * 0.3
    A = (A + A.T) / 2
    A = A + (np.abs(A).sum(axis=1) + 1.0)[:, None] * np.eye(n)

    # Convert to correlation
    D = np.sqrt(np.diag(A))
    corr = A / np.outer(D, D)
    np.fill_diagonal(corr, 1.0)

    return corr


@pytest.fixture
def two_correlation_matrices():
    """Create two correlation matrices."""
    n = 20
    np.random.seed(42)

    matrices = []
    for _ in range(2):
        A = np.random.randn(n, n) * 0.3
        A = (A + A.T) / 2
        A = A + (np.abs(A).sum(axis=1) + 1.0)[:, None] * np.eye(n)

        D = np.sqrt(np.diag(A))
        corr = A / np.outer(D, D)
        np.fill_diagonal(corr, 1.0)
        matrices.append(corr)

    return matrices[0], matrices[1]


def test_wasserstein_distance_zero(correlation_matrix):
    """Test Wasserstein distance to itself is zero."""
    dist = wasserstein_spd(correlation_matrix, correlation_matrix, metric='frobenius')
    assert abs(dist) < 1e-10


def test_wasserstein_distance_positive(two_correlation_matrices):
    """Test Wasserstein distance is positive for different matrices."""
    C1, C2 = two_correlation_matrices
    dist = wasserstein_spd(C1, C2, metric='frobenius')
    assert dist > 0


def test_wasserstein_distance_symmetric(two_correlation_matrices):
    """Test Wasserstein distance is symmetric."""
    C1, C2 = two_correlation_matrices

    dist_12 = wasserstein_spd(C1, C2, metric='frobenius')
    dist_21 = wasserstein_spd(C2, C1, metric='frobenius')

    assert abs(dist_12 - dist_21) < 1e-10


def test_wasserstein_metrics_consistent(two_correlation_matrices):
    """Test different metrics give positive distances."""
    C1, C2 = two_correlation_matrices

    dist_frob = wasserstein_spd(C1, C2, metric='frobenius')
    dist_riem = wasserstein_spd(C1, C2, metric='riemannian')
    dist_bures = wasserstein_spd(C1, C2, metric='bures')

    assert dist_frob > 0
    assert dist_riem > 0
    assert dist_bures > 0


def test_barycenter_single_matrix(correlation_matrix):
    """Test barycenter of single matrix is itself."""
    matrices = correlation_matrix[None, :, :]
    barycenter, converged = wasserstein_barycenter_spd(matrices, metric='frobenius')

    assert converged
    np.testing.assert_allclose(barycenter, correlation_matrix, rtol=1e-6)


def test_barycenter_equal_weights(two_correlation_matrices):
    """Test barycenter with equal weights."""
    C1, C2 = two_correlation_matrices
    matrices = np.stack([C1, C2])

    barycenter, converged = wasserstein_barycenter_spd(
        matrices, weights=np.array([0.5, 0.5]), metric='frobenius'
    )

    assert converged

    # Barycenter should be SPD
    eigvals = np.linalg.eigvalsh(barycenter)
    assert np.all(eigvals > -1e-10)

    # Barycenter should be between the two matrices
    dist_1 = wasserstein_spd(barycenter, C1, metric='frobenius')
    dist_2 = wasserstein_spd(barycenter, C2, metric='frobenius')
    dist_12 = wasserstein_spd(C1, C2, metric='frobenius')

    # Barycenter should be closer to each matrix than they are to each other
    assert dist_1 < dist_12
    assert dist_2 < dist_12


def test_barycenter_weighted(two_correlation_matrices):
    """Test weighted barycenter."""
    C1, C2 = two_correlation_matrices
    matrices = np.stack([C1, C2])

    # Heavily weighted toward C1
    barycenter_1, _ = wasserstein_barycenter_spd(
        matrices, weights=np.array([0.9, 0.1]), metric='frobenius'
    )

    # Heavily weighted toward C2
    barycenter_2, _ = wasserstein_barycenter_spd(
        matrices, weights=np.array([0.1, 0.9]), metric='frobenius'
    )

    # First barycenter should be closer to C1
    dist_b1_c1 = wasserstein_spd(barycenter_1, C1, metric='frobenius')
    dist_b1_c2 = wasserstein_spd(barycenter_1, C2, metric='frobenius')
    assert dist_b1_c1 < dist_b1_c2

    # Second barycenter should be closer to C2
    dist_b2_c1 = wasserstein_spd(barycenter_2, C1, metric='frobenius')
    dist_b2_c2 = wasserstein_spd(barycenter_2, C2, metric='frobenius')
    assert dist_b2_c2 < dist_b2_c1


def test_distance_matrix_shape():
    """Test distance matrix has correct shape."""
    n_matrices = 5
    n_regions = 15

    matrices = []
    for i in range(n_matrices):
        A = np.random.randn(n_regions, n_regions) * 0.3
        A = A @ A.T + np.eye(n_regions)
        matrices.append(A)

    matrices = np.array(matrices)

    dist_matrix = wasserstein_distance_matrix(matrices, metric='frobenius')

    assert dist_matrix.shape == (n_matrices, n_matrices)

    # Check symmetry
    np.testing.assert_allclose(dist_matrix, dist_matrix.T, rtol=1e-10)

    # Check diagonal is zero
    np.testing.assert_allclose(np.diag(dist_matrix), 0, atol=1e-10)


def test_trajectory_distance_euclidean():
    """Test trajectory distance with Euclidean alignment."""
    n_steps = 10
    n_regions = 10

    # Create two similar trajectories
    traj1 = [np.eye(n_regions) + np.random.randn(n_regions, n_regions) * 0.1
             for _ in range(n_steps)]
    traj2 = [np.eye(n_regions) + np.random.randn(n_regions, n_regions) * 0.1
             for _ in range(n_steps)]

    # Make them SPD
    traj1 = np.array([t @ t.T for t in traj1])
    traj2 = np.array([t @ t.T for t in traj2])

    dist = wasserstein_trajectory_distance(
        traj1, traj2, metric='frobenius', alignment='euclidean'
    )

    assert dist > 0
    assert np.isfinite(dist)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
