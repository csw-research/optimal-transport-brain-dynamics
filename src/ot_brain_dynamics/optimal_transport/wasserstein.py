"""
Wasserstein distance and optimal transport computations for SPD matrices.

This module implements Wasserstein-2 distance, barycenters, and optimal
transport plans for brain connectivity matrices.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple, List
import warnings

from ot_brain_dynamics.optimal_transport.spd_geometry import (
    sqrtm_spd,
    logm_spd,
    expm_spd,
    frechet_mean_spd,
)


def wasserstein_spd(
    C1: np.ndarray,
    C2: np.ndarray,
    metric: str = "frobenius",
    check_finite: bool = True
) -> float:
    """
    Compute Wasserstein-2 distance between two SPD matrices.

    For SPD matrices representing covariance structures, this computes:
        W_2(C1, C2) = ||C1^{1/2} - C2^{1/2}||_F  (Frobenius metric)
    or
        W_2(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F  (Riemannian metric)

    Parameters
    ----------
    C1 : np.ndarray, shape (n, n)
        First SPD matrix (correlation/covariance matrix)
    C2 : np.ndarray, shape (n, n)
        Second SPD matrix
    metric : str, default="frobenius"
        Distance metric to use:
        - "frobenius": Frobenius norm on matrix square roots
        - "riemannian": Affine-invariant Riemannian distance
        - "bures": Bures-Wasserstein distance (for Gaussian distributions)
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    float
        Wasserstein-2 distance between the two matrices

    Notes
    -----
    The Frobenius metric on matrix square roots is computationally efficient
    and provides an upper bound on the Riemannian distance. It's particularly
    useful for correlation matrices in brain connectivity analysis.

    The Bures-Wasserstein distance for zero-mean Gaussian distributions with
    covariances C1, C2 is:
        W_2^2 = Tr(C1 + C2 - 2(C1^{1/2} C2 C1^{1/2})^{1/2})

    References
    ----------
    Bhatia, R., Jain, T., & Lim, Y. (2019). On the Bures-Wasserstein distance
    between positive definite matrices. Expositiones Mathematicae, 37(2), 165-191.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"Matrices must have same shape, got {C1.shape} and {C2.shape}")

    if metric == "frobenius":
        # Compute matrix square roots
        C1_sqrt = sqrtm_spd(C1, check_finite=check_finite)
        C2_sqrt = sqrtm_spd(C2, check_finite=check_finite)

        # Frobenius norm of difference
        dist = np.linalg.norm(C1_sqrt - C2_sqrt, 'fro')

    elif metric == "riemannian":
        # Riemannian distance using log map
        from ot_brain_dynamics.optimal_transport.spd_geometry import distance_riemannian
        dist = distance_riemannian(C1, C2, check_finite=check_finite)

    elif metric == "bures":
        # Bures-Wasserstein distance
        C1_sqrt = sqrtm_spd(C1, check_finite=check_finite)

        # Compute (C1^{1/2} C2 C1^{1/2})^{1/2}
        M = C1_sqrt @ C2 @ C1_sqrt
        M_sqrt = sqrtm_spd(M, check_finite=check_finite)

        # W_2^2 = Tr(C1 + C2 - 2 M^{1/2})
        trace_sum = np.trace(C1) + np.trace(C2) - 2 * np.trace(M_sqrt)
        dist = np.sqrt(max(0, trace_sum))  # Ensure non-negative due to numerical errors

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'frobenius', 'riemannian', or 'bures'")

    return float(dist)


def wasserstein_barycenter_spd(
    matrices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    metric: str = "frobenius",
    max_iter: int = 100,
    tol: float = 1e-6,
    check_finite: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Compute Wasserstein barycenter (Fréchet mean) of SPD matrices.

    The barycenter minimizes the weighted sum of squared Wasserstein distances:
        C* = argmin_C sum_i w_i W_2(C, C_i)^2

    Parameters
    ----------
    matrices : np.ndarray, shape (m, n, n)
        Array of m SPD matrices
    weights : np.ndarray, shape (m,), optional
        Weights for each matrix (default: uniform)
    metric : str, default="frobenius"
        Wasserstein metric to use (see wasserstein_spd)
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    barycenter : np.ndarray, shape (n, n)
        Wasserstein barycenter of input matrices
    converged : bool
        Whether algorithm converged

    Notes
    -----
    For the Frobenius metric, the barycenter has a closed-form solution:
        C* = (sum_i w_i C_i^{1/2})^2

    For Riemannian metric, uses gradient descent on the manifold.

    References
    ----------
    Agueh, M., & Carlier, G. (2011). Barycenters in the Wasserstein space.
    SIAM Journal on Mathematical Analysis, 43(2), 904-924.
    """
    m, n, _ = matrices.shape

    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.asarray(weights)
        if len(weights) != m:
            raise ValueError(f"weights must have length {m}, got {len(weights)}")
        weights = weights / weights.sum()

    if metric == "frobenius":
        # Closed-form solution for Frobenius metric
        sqrt_matrices = np.array([sqrtm_spd(C, check_finite=check_finite) for C in matrices])
        weighted_sum = np.sum(weights[:, None, None] * sqrt_matrices, axis=0)

        # Barycenter is the square of the weighted sum
        barycenter = weighted_sum @ weighted_sum
        converged = True

    elif metric in ["riemannian", "bures"]:
        # Use Fréchet mean algorithm
        barycenter, converged = frechet_mean_spd(
            matrices, weights, max_iter=max_iter, tol=tol, check_finite=check_finite
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return barycenter, converged


def optimal_transport_plan(
    C1: np.ndarray,
    C2: np.ndarray,
    reg: float = 0.01,
    check_finite: bool = True
) -> np.ndarray:
    """
    Compute optimal transport plan between two SPD matrices.

    The transport plan describes how to optimally "move mass" from C1 to C2,
    which in brain connectivity terms represents how network connections
    reorganize between states.

    Parameters
    ----------
    C1 : np.ndarray, shape (n, n)
        Source SPD matrix
    C2 : np.ndarray, shape (n, n)
        Target SPD matrix
    reg : float, default=0.01
        Entropic regularization parameter (lower = more accurate but slower)
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    plan : np.ndarray, shape (n, n)
        Optimal transport plan matrix where plan[i,j] represents the
        "amount" transported from feature i in C1 to feature j in C2

    Notes
    -----
    This uses entropic regularization (Sinkhorn algorithm) for computational
    efficiency. The transport plan can reveal which brain regions/connections
    in one state correspond to regions/connections in another state.

    The plan satisfies:
        - Non-negativity: plan[i,j] >= 0
        - Marginal constraints based on matrix structure

    For correlation matrices, we interpret the marginals as related to the
    variance structure of each region.

    References
    ----------
    Peyré, G., & Cuturi, M. (2019). Computational optimal transport.
    Foundations and Trends in Machine Learning, 11(5-6), 355-607.
    """
    n = C1.shape[0]

    # Extract marginals from matrix structure
    # For covariance/correlation matrices, use diagonal elements (variances)
    # and off-diagonal structure
    marginal_1 = np.diag(C1)
    marginal_2 = np.diag(C2)

    # Normalize to probability distributions
    marginal_1 = marginal_1 / marginal_1.sum()
    marginal_2 = marginal_2 / marginal_2.sum()

    # Compute cost matrix based on geodesic distances
    # For each pair (i,j), cost reflects how different regions i and j are
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Cost based on correlation difference and geometric distance
            # This is a heuristic cost matrix design
            corr_diff = abs(C1[i, i] - C2[j, j])
            cost[i, j] = corr_diff

    # Solve optimal transport with Sinkhorn
    from ot_brain_dynamics.optimal_transport.sinkhorn import sinkhorn_plan

    plan = sinkhorn_plan(
        marginal_1, marginal_2, cost, reg=reg, max_iter=1000, tol=1e-9
    )

    return plan


def wasserstein_distance_matrix(
    matrices: np.ndarray,
    metric: str = "frobenius",
    check_finite: bool = True
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distance matrix for multiple SPD matrices.

    Useful for analyzing trajectories of brain states over time or
    comparing multiple network configurations.

    Parameters
    ----------
    matrices : np.ndarray, shape (m, n, n)
        Array of m SPD matrices
    metric : str, default="frobenius"
        Wasserstein metric to use
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    distances : np.ndarray, shape (m, m)
        Symmetric matrix of pairwise Wasserstein distances

    Examples
    --------
    >>> # Compute distances between connectivity states over time
    >>> conn_matrices = sliding_window_connectivity(fmri_data)
    >>> dist_matrix = wasserstein_distance_matrix(conn_matrices)
    >>> # dist_matrix[i,j] = distance between time windows i and j
    """
    m = len(matrices)
    distances = np.zeros((m, m))

    for i in range(m):
        for j in range(i + 1, m):
            dist = wasserstein_spd(
                matrices[i], matrices[j], metric=metric, check_finite=check_finite
            )
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def wasserstein_trajectory_distance(
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    metric: str = "frobenius",
    alignment: str = "dtw",
    check_finite: bool = True
) -> float:
    """
    Compute distance between two trajectories of SPD matrices.

    Useful for comparing brain state evolution across different subjects,
    tasks, or experimental conditions.

    Parameters
    ----------
    trajectory1 : np.ndarray, shape (t1, n, n)
        First trajectory of t1 SPD matrices
    trajectory2 : np.ndarray, shape (t2, n, n)
        Second trajectory of t2 SPD matrices
    metric : str, default="frobenius"
        Wasserstein metric for point-wise distances
    alignment : str, default="dtw"
        How to align trajectories:
        - "dtw": Dynamic time warping
        - "euclidean": Direct alignment (requires same length)
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    float
        Distance between the two trajectories

    Notes
    -----
    Dynamic time warping finds optimal non-linear alignment between
    trajectories, useful when brain state transitions occur at different
    rates across subjects/conditions.
    """
    t1, n, _ = trajectory1.shape
    t2, _, _ = trajectory2.shape

    if alignment == "euclidean":
        if t1 != t2:
            raise ValueError(
                f"Trajectories must have same length for Euclidean alignment, "
                f"got {t1} and {t2}"
            )

        total_dist = 0.0
        for i in range(t1):
            dist = wasserstein_spd(
                trajectory1[i], trajectory2[i], metric=metric, check_finite=check_finite
            )
            total_dist += dist ** 2

        return np.sqrt(total_dist)

    elif alignment == "dtw":
        # Dynamic time warping
        # Compute pairwise distance matrix
        cost_matrix = np.zeros((t1, t2))
        for i in range(t1):
            for j in range(t2):
                cost_matrix[i, j] = wasserstein_spd(
                    trajectory1[i], trajectory2[j], metric=metric, check_finite=check_finite
                )

        # DTW algorithm
        dtw_matrix = np.full((t1 + 1, t2 + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, t1 + 1):
            for j in range(1, t2 + 1):
                cost = cost_matrix[i - 1, j - 1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1]   # match
                )

        return dtw_matrix[t1, t2]

    else:
        raise ValueError(f"Unknown alignment method: {alignment}")
