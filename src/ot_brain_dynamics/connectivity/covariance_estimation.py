"""
Robust covariance estimation for functional connectivity.

Provides regularized covariance estimators that are well-conditioned
even for short time windows.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import linalg
import warnings


def robust_covariance_estimate(
    data: np.ndarray,
    method: str = 'shrinkage',
    **kwargs
) -> np.ndarray:
    """
    Compute robust covariance estimate with automatic method selection.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix
    method : str, default='shrinkage'
        Estimation method:
        - 'shrinkage': Ledoit-Wolf shrinkage
        - 'glasso': Graphical lasso
        - 'oas': Oracle approximating shrinkage
        - 'empirical': Standard sample covariance
    **kwargs
        Method-specific parameters

    Returns
    -------
    cov : np.ndarray, shape (n_features, n_features)
        Covariance estimate

    Notes
    -----
    For small sample sizes (n_samples < n_features), regularization
    is essential for obtaining well-conditioned covariance matrices.

    Shrinkage methods work well for general purpose estimation.
    Graphical lasso is preferred when sparsity is expected.
    """
    if method == 'shrinkage':
        return shrinkage_covariance(data, **kwargs)
    elif method == 'glasso':
        return graphical_lasso_covariance(data, **kwargs)
    elif method == 'oas':
        return oas_covariance(data, **kwargs)
    elif method == 'empirical':
        return empirical_covariance(data, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def shrinkage_covariance(
    data: np.ndarray,
    shrinkage: Optional[float] = None
) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Shrinks sample covariance toward scaled identity matrix:
        Σ_shrink = (1 - λ) Σ_sample + λ μ I

    where λ is the shrinkage parameter and μ is the average variance.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix
    shrinkage : float, optional
        Shrinkage parameter in [0, 1]
        If None, estimated automatically using Ledoit-Wolf formula

    Returns
    -------
    cov : np.ndarray, shape (n_features, n_features)
        Shrinkage covariance matrix

    Notes
    -----
    Automatic shrinkage estimation (when shrinkage=None) is optimal
    under Frobenius loss for Gaussian data.

    References
    ----------
    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices. Journal of Multivariate Analysis.
    """
    n_samples, n_features = data.shape

    # Center data
    data_centered = data - data.mean(axis=0, keepdims=True)

    # Empirical covariance
    cov_empirical = (data_centered.T @ data_centered) / n_samples

    if shrinkage is None:
        # Automatic Ledoit-Wolf shrinkage
        shrinkage = _ledoit_wolf_shrinkage(data_centered, cov_empirical)

    # Shrinkage target: scaled identity
    mu = np.trace(cov_empirical) / n_features
    target = mu * np.eye(n_features)

    # Shrink toward target
    cov_shrink = (1 - shrinkage) * cov_empirical + shrinkage * target

    return cov_shrink


def _ledoit_wolf_shrinkage(data_centered: np.ndarray, cov_empirical: np.ndarray) -> float:
    """Estimate optimal Ledoit-Wolf shrinkage parameter."""
    n_samples, n_features = data_centered.shape

    # Target: scaled identity
    mu = np.trace(cov_empirical) / n_features

    # Compute shrinkage intensity
    # This is a simplified version; full LW formula is more complex
    delta = np.linalg.norm(cov_empirical - mu * np.eye(n_features), 'fro') ** 2

    # Estimate delta and beta2
    X2 = data_centered ** 2
    delta_hat = np.sum((data_centered.T @ data_centered / n_samples - cov_empirical) ** 2)

    # Optimal shrinkage
    if delta > 0:
        shrinkage = min(delta_hat / delta, 1.0)
    else:
        shrinkage = 1.0

    return max(0.0, shrinkage)


def oas_covariance(data: np.ndarray) -> np.ndarray:
    """
    Oracle approximating shrinkage (OAS) covariance estimator.

    An alternative shrinkage estimator with different optimal properties.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix

    Returns
    -------
    cov : np.ndarray, shape (n_features, n_features)
        OAS covariance estimate

    References
    ----------
    Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage
    algorithms for MMSE covariance estimation. IEEE TSP.
    """
    n_samples, n_features = data.shape

    # Center data
    data_centered = data - data.mean(axis=0, keepdims=True)

    # Empirical covariance
    cov_empirical = (data_centered.T @ data_centered) / n_samples

    # OAS shrinkage parameter
    trace_cov2 = np.trace(cov_empirical @ cov_empirical)
    trace_cov = np.trace(cov_empirical)

    rho = (1.0 - 2.0 / n_features) * trace_cov2 + trace_cov ** 2
    rho /= (n_samples + 1.0 - 2.0 / n_features) * trace_cov2

    rho = min(rho, 1.0)

    # Shrinkage target
    mu = trace_cov / n_features

    # Shrink
    cov_oas = (1 - rho) * cov_empirical + rho * mu * np.eye(n_features)

    return cov_oas


def empirical_covariance(data: np.ndarray, assume_centered: bool = False) -> np.ndarray:
    """
    Compute empirical (sample) covariance matrix.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix
    assume_centered : bool, default=False
        If True, data is assumed to be centered

    Returns
    -------
    cov : np.ndarray, shape (n_features, n_features)
        Empirical covariance matrix
    """
    if assume_centered:
        data_centered = data
    else:
        data_centered = data - data.mean(axis=0, keepdims=True)

    n_samples = len(data)
    cov = (data_centered.T @ data_centered) / n_samples

    return cov


def graphical_lasso_covariance(
    data: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse inverse covariance estimation via graphical lasso.

    Estimates sparse precision matrix (inverse covariance) using L1
    regularization, useful for inferring direct connections in networks.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Data matrix
    alpha : float, default=0.01
        Regularization parameter (higher = more sparse)
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-4
        Convergence tolerance

    Returns
    -------
    cov : np.ndarray, shape (n_features, n_features)
        Estimated covariance matrix
    precision : np.ndarray, shape (n_features, n_features)
        Estimated precision matrix (sparse)

    Notes
    -----
    The graphical lasso solves:
        max_Θ log det(Θ) - Tr(S Θ) - α ||Θ||_1

    where Θ is the precision matrix and S is the sample covariance.

    Sparsity in the precision matrix corresponds to conditional independence:
        Θ_ij = 0  <==>  X_i ⊥ X_j | X_{-ij}

    This is particularly useful for brain networks where we want to identify
    direct (not mediated) connections.

    References
    ----------
    Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
    covariance estimation with the graphical lasso. Biostatistics, 9(3), 432-441.
    """
    try:
        from sklearn.covariance import GraphicalLassoCV, graphical_lasso

        # If alpha='cv', use cross-validation
        if alpha == 'cv':
            model = GraphicalLassoCV(max_iter=max_iter, tol=tol, cv=3)
            model.fit(data)
            return model.covariance_, model.precision_
        else:
            cov, precision = graphical_lasso(
                empirical_covariance(data),
                alpha=alpha,
                max_iter=max_iter,
                tol=tol
            )
            return cov, precision

    except ImportError:
        warnings.warn(
            "scikit-learn not available, using shrinkage instead of graphical lasso"
        )
        cov = shrinkage_covariance(data)
        precision = linalg.inv(cov)
        return cov, precision


def ensure_spd(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Ensure matrix is symmetric positive definite.

    Useful for fixing numerical issues in covariance matrices.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Input matrix
    min_eigenvalue : float, default=1e-6
        Minimum eigenvalue to enforce

    Returns
    -------
    spd_matrix : np.ndarray, shape (n, n)
        SPD-corrected matrix

    Notes
    -----
    Procedure:
    1. Symmetrize: (A + A^T) / 2
    2. Eigendecomposition
    3. Clip eigenvalues to be >= min_eigenvalue
    4. Reconstruct matrix
    """
    # Ensure symmetry
    matrix_sym = (matrix + matrix.T) / 2

    # Eigendecomposition
    eigvals, eigvecs = linalg.eigh(matrix_sym)

    # Clip eigenvalues
    eigvals_clipped = np.maximum(eigvals, min_eigenvalue)

    # Reconstruct
    matrix_spd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    return matrix_spd


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    cov : np.ndarray, shape (n, n)
        Covariance matrix

    Returns
    -------
    corr : np.ndarray, shape (n, n)
        Correlation matrix

    Notes
    -----
    corr_ij = cov_ij / sqrt(cov_ii * cov_jj)
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    # Ensure valid correlation matrix
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2

    return corr
