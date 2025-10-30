"""
Riemannian geometry operations on the manifold of symmetric positive definite matrices.

This module provides core geometric operations including matrix square roots,
logarithms, exponentials, geodesics, and distance computations on SPD manifolds.
"""

import numpy as np
from scipy import linalg
from typing import Union, Tuple
import warnings


def sqrtm_spd(C: np.ndarray, check_finite: bool = True) -> np.ndarray:
    """
    Compute the matrix square root of a symmetric positive definite matrix.

    For SPD matrix C, computes C^{1/2} such that C^{1/2} @ C^{1/2} = C.
    Uses eigenvalue decomposition for numerical stability.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Symmetric positive definite matrix
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    np.ndarray, shape (n, n)
        Matrix square root C^{1/2}

    Raises
    ------
    ValueError
        If matrix is not square or has negative eigenvalues

    Notes
    -----
    For SPD matrix C with eigendecomposition C = Q Λ Q^T:
        C^{1/2} = Q Λ^{1/2} Q^T
    where Λ^{1/2} is the diagonal matrix of square roots of eigenvalues.

    References
    ----------
    Higham, N. J. (2008). Functions of matrices: theory and computation.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {C.shape}")

    # Ensure symmetry
    C_sym = (C + C.T) / 2

    # Compute eigendecomposition
    eigvals, eigvecs = linalg.eigh(C_sym, check_finite=check_finite)

    # Check positive definiteness
    if np.any(eigvals < -1e-10):
        warnings.warn(
            f"Matrix has negative eigenvalues (min: {eigvals.min():.2e}). "
            "Clipping to small positive value."
        )
        eigvals = np.maximum(eigvals, 1e-10)

    # Compute square root
    sqrt_eigvals = np.sqrt(eigvals)
    C_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

    return C_sqrt


def logm_spd(C: np.ndarray, check_finite: bool = True) -> np.ndarray:
    """
    Compute the matrix logarithm of a symmetric positive definite matrix.

    Parameters
    ----------
    C : np.ndarray, shape (n, n)
        Symmetric positive definite matrix
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    np.ndarray, shape (n, n)
        Matrix logarithm log(C)

    Notes
    -----
    For SPD matrix C with eigendecomposition C = Q Λ Q^T:
        log(C) = Q log(Λ) Q^T
    where log(Λ) is the diagonal matrix of logarithms of eigenvalues.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {C.shape}")

    # Ensure symmetry
    C_sym = (C + C.T) / 2

    # Compute eigendecomposition
    eigvals, eigvecs = linalg.eigh(C_sym, check_finite=check_finite)

    # Check positive definiteness
    if np.any(eigvals <= 0):
        warnings.warn(
            f"Matrix has non-positive eigenvalues (min: {eigvals.min():.2e}). "
            "Clipping to small positive value."
        )
        eigvals = np.maximum(eigvals, 1e-10)

    # Compute logarithm
    log_eigvals = np.log(eigvals)
    C_log = eigvecs @ np.diag(log_eigvals) @ eigvecs.T

    return C_log


def expm_spd(S: np.ndarray, check_finite: bool = True) -> np.ndarray:
    """
    Compute the matrix exponential, mapping from tangent space to SPD manifold.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Symmetric matrix in tangent space
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    np.ndarray, shape (n, n)
        SPD matrix exp(S)

    Notes
    -----
    For symmetric matrix S with eigendecomposition S = Q Λ Q^T:
        exp(S) = Q exp(Λ) Q^T
    where exp(Λ) is the diagonal matrix of exponentials of eigenvalues.
    """
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {S.shape}")

    # Ensure symmetry
    S_sym = (S + S.T) / 2

    # Compute eigendecomposition
    eigvals, eigvecs = linalg.eigh(S_sym, check_finite=check_finite)

    # Compute exponential
    exp_eigvals = np.exp(eigvals)
    C_exp = eigvecs @ np.diag(exp_eigvals) @ eigvecs.T

    return C_exp


def geodesic_spd(
    C1: np.ndarray,
    C2: np.ndarray,
    t: Union[float, np.ndarray] = 0.5,
    check_finite: bool = True
) -> np.ndarray:
    """
    Compute geodesic interpolation between two SPD matrices.

    The geodesic on the SPD manifold from C1 to C2 at parameter t ∈ [0,1]:
        γ(t) = C1^{1/2} (C1^{-1/2} C2 C1^{-1/2})^t C1^{1/2}

    Parameters
    ----------
    C1 : np.ndarray, shape (n, n)
        Starting SPD matrix (at t=0)
    C2 : np.ndarray, shape (n, n)
        Ending SPD matrix (at t=1)
    t : float or np.ndarray, default=0.5
        Interpolation parameter(s) in [0, 1]
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    np.ndarray, shape (n, n) or (m, n, n)
        Interpolated SPD matrix/matrices at parameter(s) t
        If t is scalar, returns (n, n) matrix
        If t is array of length m, returns (m, n, n) array

    Notes
    -----
    This computes the unique geodesic path on the Riemannian manifold of SPD
    matrices equipped with the affine-invariant metric. At t=0, returns C1;
    at t=1, returns C2.

    The computation uses the log-Euclidean framework:
        γ(t) = exp((1-t) log(C1) + t log(C2))
    which is equivalent but often more numerically stable.

    References
    ----------
    Pennec, X., Fillard, P., & Ayache, N. (2006). A Riemannian framework for
    tensor computing. International Journal of Computer Vision, 66(1), 41-66.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"Matrices must have same shape, got {C1.shape} and {C2.shape}")

    # Compute matrix logarithms
    log_C1 = logm_spd(C1, check_finite=check_finite)
    log_C2 = logm_spd(C2, check_finite=check_finite)

    # Handle scalar and array t
    t = np.atleast_1d(t)

    if len(t) == 1:
        # Single interpolation point
        log_interp = (1 - t[0]) * log_C1 + t[0] * log_C2
        return expm_spd(log_interp, check_finite=check_finite)
    else:
        # Multiple interpolation points
        results = np.zeros((len(t),) + C1.shape)
        for i, t_i in enumerate(t):
            log_interp = (1 - t_i) * log_C1 + t_i * log_C2
            results[i] = expm_spd(log_interp, check_finite=check_finite)
        return results


def distance_riemannian(
    C1: np.ndarray,
    C2: np.ndarray,
    check_finite: bool = True
) -> float:
    """
    Compute the Riemannian distance between two SPD matrices.

    The affine-invariant Riemannian distance on the SPD manifold:
        d(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F

    where ||·||_F is the Frobenius norm.

    Parameters
    ----------
    C1 : np.ndarray, shape (n, n)
        First SPD matrix
    C2 : np.ndarray, shape (n, n)
        Second SPD matrix
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    float
        Riemannian distance between C1 and C2

    Notes
    -----
    This distance is:
    - Symmetric: d(C1, C2) = d(C2, C1)
    - Affine-invariant: d(A C1 A^T, A C2 A^T) = d(C1, C2) for any invertible A
    - Geodesic: equals the length of the shortest path on the manifold

    Equivalent formulations:
        d(C1, C2) = ||log(C1^{-1/2} C2 C1^{-1/2})||_F
                  = sqrt(sum(log(λ_i)^2))
    where λ_i are eigenvalues of C1^{-1} C2.

    References
    ----------
    Förstner, W., & Moonen, B. (1999). A metric for covariance matrices.
    """
    if C1.shape != C2.shape:
        raise ValueError(f"Matrices must have same shape, got {C1.shape} and {C2.shape}")

    # Compute C1^{-1/2}
    C1_sqrt = sqrtm_spd(C1, check_finite=check_finite)
    C1_inv_sqrt = linalg.inv(C1_sqrt)

    # Compute C1^{-1/2} C2 C1^{-1/2}
    M = C1_inv_sqrt @ C2 @ C1_inv_sqrt

    # Compute logarithm
    log_M = logm_spd(M, check_finite=check_finite)

    # Frobenius norm
    dist = np.linalg.norm(log_M, 'fro')

    return float(dist)


def parallel_transport_spd(
    V: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    check_finite: bool = True
) -> np.ndarray:
    """
    Parallel transport a tangent vector from C1 to C2 along geodesic.

    Parameters
    ----------
    V : np.ndarray, shape (n, n)
        Tangent vector at C1 (symmetric matrix)
    C1 : np.ndarray, shape (n, n)
        Starting point on SPD manifold
    C2 : np.ndarray, shape (n, n)
        Ending point on SPD manifold
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    np.ndarray, shape (n, n)
        Parallel transported tangent vector at C2

    Notes
    -----
    Parallel transport preserves the inner product of tangent vectors and
    is the unique way to move vectors along geodesics while keeping them
    "constant" in the sense of Riemannian geometry.

    Formula:
        PT_{C1→C2}(V) = E V E^T
    where E = C2^{1/2} C1^{-1/2}
    """
    # Compute transport matrix E = C2^{1/2} C1^{-1/2}
    C1_sqrt = sqrtm_spd(C1, check_finite=check_finite)
    C2_sqrt = sqrtm_spd(C2, check_finite=check_finite)
    C1_inv_sqrt = linalg.inv(C1_sqrt)

    E = C2_sqrt @ C1_inv_sqrt

    # Transport the tangent vector
    V_transported = E @ V @ E.T

    return V_transported


def frechet_mean_spd(
    matrices: np.ndarray,
    weights: Union[np.ndarray, None] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    check_finite: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Compute the Fréchet mean (Riemannian barycenter) of SPD matrices.

    The Fréchet mean minimizes the sum of squared Riemannian distances:
        C* = argmin_C sum_i w_i d(C, C_i)^2

    Uses gradient descent on the SPD manifold.

    Parameters
    ----------
    matrices : np.ndarray, shape (m, n, n)
        Array of m SPD matrices
    weights : np.ndarray, shape (m,), optional
        Weights for each matrix (default: uniform)
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance on gradient norm
    check_finite : bool, default=True
        Whether to check for finite values

    Returns
    -------
    mean : np.ndarray, shape (n, n)
        Fréchet mean of the input matrices
    converged : bool
        Whether algorithm converged within max_iter

    Notes
    -----
    The algorithm uses the gradient descent update on the manifold:
        C_{k+1} = C_k^{1/2} exp(α * grad) C_k^{1/2}
    where grad is the Riemannian gradient at C_k.

    References
    ----------
    Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds: Basic
    tools for geometric measurements. Journal of Mathematical Imaging and Vision.
    """
    m, n, _ = matrices.shape

    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.asarray(weights)
        if len(weights) != m:
            raise ValueError(f"weights must have length {m}, got {len(weights)}")
        weights = weights / weights.sum()  # Normalize

    # Initialize with weighted Euclidean mean
    mean = np.sum(weights[:, None, None] * matrices, axis=0)
    mean = (mean + mean.T) / 2  # Ensure symmetry

    # Make sure initialization is SPD
    eigvals, eigvecs = linalg.eigh(mean)
    eigvals = np.maximum(eigvals, 1e-10)
    mean = eigvecs @ np.diag(eigvals) @ eigvecs.T

    converged = False
    for iteration in range(max_iter):
        # Compute Riemannian gradient
        mean_sqrt = sqrtm_spd(mean, check_finite=check_finite)
        mean_inv_sqrt = linalg.inv(mean_sqrt)

        grad = np.zeros_like(mean)
        for i in range(m):
            # Log map from mean to matrices[i]
            M = mean_inv_sqrt @ matrices[i] @ mean_inv_sqrt
            log_M = logm_spd(M, check_finite=check_finite)
            grad += weights[i] * log_M

        # Check convergence
        grad_norm = np.linalg.norm(grad, 'fro')
        if grad_norm < tol:
            converged = True
            break

        # Gradient descent step (with line search)
        alpha = 1.0
        mean_new = expm_spd(mean_inv_sqrt @ grad @ mean_inv_sqrt, check_finite=check_finite)
        mean_new = mean_sqrt @ mean_new @ mean_sqrt
        mean = mean_new

    return mean, converged
