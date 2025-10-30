"""Tests for SPD matrix geometry operations."""

import numpy as np
import pytest
from scipy import linalg

from ot_brain_dynamics.optimal_transport.spd_geometry import (
    sqrtm_spd,
    logm_spd,
    expm_spd,
    geodesic_spd,
    distance_riemannian,
    frechet_mean_spd,
)


@pytest.fixture
def spd_matrix():
    """Create a simple SPD matrix."""
    n = 10
    A = np.random.randn(n, n)
    return A @ A.T + np.eye(n)


@pytest.fixture
def two_spd_matrices():
    """Create two SPD matrices."""
    n = 10
    A1 = np.random.randn(n, n)
    A2 = np.random.randn(n, n)
    C1 = A1 @ A1.T + np.eye(n)
    C2 = A2 @ A2.T + np.eye(n)
    return C1, C2


def test_sqrtm_spd_basic(spd_matrix):
    """Test basic properties of matrix square root."""
    C_sqrt = sqrtm_spd(spd_matrix)

    # Square root squared should give original matrix
    C_reconstructed = C_sqrt @ C_sqrt
    np.testing.assert_allclose(C_reconstructed, spd_matrix, rtol=1e-10)

    # Square root should be symmetric
    np.testing.assert_allclose(C_sqrt, C_sqrt.T, rtol=1e-10)

    # Square root should be PSD
    eigvals = np.linalg.eigvalsh(C_sqrt)
    assert np.all(eigvals >= -1e-10)


def test_logm_expm_inverse(spd_matrix):
    """Test that log and exp are inverses."""
    log_C = logm_spd(spd_matrix)
    C_reconstructed = expm_spd(log_C)

    np.testing.assert_allclose(C_reconstructed, spd_matrix, rtol=1e-8)


def test_geodesic_endpoints(two_spd_matrices):
    """Test geodesic endpoints match input matrices."""
    C1, C2 = two_spd_matrices

    # At t=0, should get C1
    gamma_0 = geodesic_spd(C1, C2, t=0.0)
    np.testing.assert_allclose(gamma_0, C1, rtol=1e-10)

    # At t=1, should get C2
    gamma_1 = geodesic_spd(C1, C2, t=1.0)
    np.testing.assert_allclose(gamma_1, C2, rtol=1e-10)


def test_geodesic_symmetry(two_spd_matrices):
    """Test geodesic symmetry property."""
    C1, C2 = two_spd_matrices

    # Geodesic from C1 to C2 at t
    gamma_forward = geodesic_spd(C1, C2, t=0.3)

    # Should equal geodesic from C2 to C1 at 1-t
    gamma_backward = geodesic_spd(C2, C1, t=0.7)

    np.testing.assert_allclose(gamma_forward, gamma_backward, rtol=1e-8)


def test_distance_symmetry(two_spd_matrices):
    """Test distance is symmetric."""
    C1, C2 = two_spd_matrices

    dist_12 = distance_riemannian(C1, C2)
    dist_21 = distance_riemannian(C2, C1)

    assert abs(dist_12 - dist_21) < 1e-10


def test_distance_triangle_inequality(spd_matrix):
    """Test triangle inequality for Riemannian distance."""
    n = spd_matrix.shape[0]

    # Create three matrices
    A1 = np.random.randn(n, n)
    A2 = np.random.randn(n, n)
    A3 = np.random.randn(n, n)

    C1 = A1 @ A1.T + np.eye(n)
    C2 = A2 @ A2.T + np.eye(n)
    C3 = A3 @ A3.T + np.eye(n)

    dist_12 = distance_riemannian(C1, C2)
    dist_23 = distance_riemannian(C2, C3)
    dist_13 = distance_riemannian(C1, C3)

    # Triangle inequality: d(C1, C3) <= d(C1, C2) + d(C2, C3)
    assert dist_13 <= dist_12 + dist_23 + 1e-8


def test_frechet_mean_single_matrix(spd_matrix):
    """Test Fréchet mean of single matrix is itself."""
    matrices = spd_matrix[None, :, :]
    mean, converged = frechet_mean_spd(matrices)

    assert converged
    np.testing.assert_allclose(mean, spd_matrix, rtol=1e-6)


def test_frechet_mean_identical_matrices(spd_matrix):
    """Test Fréchet mean of identical matrices is the matrix itself."""
    matrices = np.stack([spd_matrix] * 5)
    mean, converged = frechet_mean_spd(matrices)

    assert converged
    np.testing.assert_allclose(mean, spd_matrix, rtol=1e-5)


def test_frechet_mean_two_matrices(two_spd_matrices):
    """Test Fréchet mean of two matrices with equal weights."""
    C1, C2 = two_spd_matrices
    matrices = np.stack([C1, C2])

    mean, converged = frechet_mean_spd(matrices, weights=np.array([0.5, 0.5]))

    # Mean should be at midpoint of geodesic
    midpoint = geodesic_spd(C1, C2, t=0.5)

    np.testing.assert_allclose(mean, midpoint, rtol=1e-4)


def test_geodesic_multiple_t():
    """Test geodesic with multiple t values."""
    n = 10
    C1 = np.eye(n)
    C2 = np.eye(n) * 2

    t_values = np.linspace(0, 1, 5)
    gammas = geodesic_spd(C1, C2, t_values)

    assert gammas.shape == (5, n, n)

    # Check endpoints
    np.testing.assert_allclose(gammas[0], C1, rtol=1e-10)
    np.testing.assert_allclose(gammas[-1], C2, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
