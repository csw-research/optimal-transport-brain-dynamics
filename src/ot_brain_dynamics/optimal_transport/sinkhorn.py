"""
Sinkhorn algorithm for entropic optimal transport.

Provides efficient computations of optimal transport using entropic regularization,
with both CPU and GPU implementations.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def sinkhorn_plan(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-9,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute optimal transport plan using Sinkhorn-Knopp algorithm.

    Solves the regularized optimal transport problem:
        min_T <T, M> + reg * KL(T | a ⊗ b)
        s.t. T @ 1 = a, T^T @ 1 = b

    where M is the cost matrix and KL is Kullback-Leibler divergence.

    Parameters
    ----------
    a : np.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : np.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : np.ndarray, shape (n, m)
        Cost matrix
    reg : float, default=0.01
        Entropic regularization parameter (lower = more accurate, slower)
    max_iter : int, default=1000
        Maximum number of Sinkhorn iterations
    tol : float, default=1e-9
        Convergence tolerance on marginal errors
    use_gpu : bool, default=False
        Whether to use GPU acceleration (requires PyTorch)

    Returns
    -------
    plan : np.ndarray, shape (n, m)
        Optimal transport plan

    Notes
    -----
    The Sinkhorn algorithm iteratively projects onto marginal constraints:
        K = exp(-M / reg)
        u_{k+1} = a / (K @ v_k)
        v_{k+1} = b / (K^T @ u_{k+1})
        T = diag(u) K diag(v)

    Entropic regularization provides:
    - Faster convergence (linear vs. cubic in problem size)
    - Smooth solutions (differentiable)
    - Statistical robustness

    References
    ----------
    Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal
    transport. Advances in Neural Information Processing Systems, 26.
    """
    if not np.allclose(a.sum(), 1.0) or not np.allclose(b.sum(), 1.0):
        warnings.warn("Distributions should sum to 1, normalizing...")
        a = a / a.sum()
        b = b / b.sum()

    if use_gpu and TORCH_AVAILABLE:
        return _sinkhorn_plan_gpu(a, b, M, reg, max_iter, tol)
    else:
        return _sinkhorn_plan_cpu(a, b, M, reg, max_iter, tol)


def _sinkhorn_plan_cpu(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """CPU implementation of Sinkhorn algorithm."""
    n, m = M.shape

    # Compute kernel matrix
    K = np.exp(-M / reg)

    # Initialize dual variables
    u = np.ones(n) / n
    v = np.ones(m) / m

    # Sinkhorn iterations
    for iteration in range(max_iter):
        u_prev = u.copy()

        # Update v
        v = b / (K.T @ u)

        # Update u
        u = a / (K @ v)

        # Check convergence
        if iteration % 10 == 0:
            # Compute marginal errors
            err_a = np.abs((u[:, None] * K * v[None, :]).sum(axis=1) - a).max()
            err_b = np.abs((u[:, None] * K * v[None, :]).sum(axis=0) - b).max()
            err = max(err_a, err_b)

            if err < tol:
                break

    # Compute transport plan
    plan = u[:, None] * K * v[None, :]

    return plan


def _sinkhorn_plan_gpu(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float,
    max_iter: int,
    tol: float
) -> np.ndarray:
    """GPU implementation of Sinkhorn algorithm using PyTorch."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors
    a_t = torch.from_numpy(a).float().to(device)
    b_t = torch.from_numpy(b).float().to(device)
    M_t = torch.from_numpy(M).float().to(device)

    # Compute kernel matrix
    K = torch.exp(-M_t / reg)

    # Initialize dual variables
    u = torch.ones_like(a_t) / len(a_t)
    v = torch.ones_like(b_t) / len(b_t)

    # Sinkhorn iterations
    for iteration in range(max_iter):
        # Update v
        v = b_t / (K.t() @ u)

        # Update u
        u = a_t / (K @ v)

        # Check convergence
        if iteration % 10 == 0:
            plan_t = u[:, None] * K * v[None, :]
            err_a = torch.abs(plan_t.sum(dim=1) - a_t).max().item()
            err_b = torch.abs(plan_t.sum(dim=0) - b_t).max().item()
            err = max(err_a, err_b)

            if err < tol:
                break

    # Compute transport plan
    plan_t = u[:, None] * K * v[None, :]

    return plan_t.cpu().numpy()


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-9,
    use_gpu: bool = False
) -> float:
    """
    Compute Sinkhorn distance (regularized optimal transport cost).

    Parameters
    ----------
    a : np.ndarray, shape (n,)
        Source distribution
    b : np.ndarray, shape (m,)
        Target distribution
    M : np.ndarray, shape (n, m)
        Cost matrix
    reg : float, default=0.01
        Entropic regularization parameter
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-9
        Convergence tolerance
    use_gpu : bool, default=False
        Use GPU acceleration

    Returns
    -------
    float
        Sinkhorn distance (transport cost)

    Notes
    -----
    The Sinkhorn distance is:
        D_reg(a, b) = <T*, M> + reg * KL(T* | a ⊗ b)
    where T* is the optimal transport plan.

    As reg → 0, converges to true Wasserstein distance.
    """
    plan = sinkhorn_plan(a, b, M, reg, max_iter, tol, use_gpu)
    distance = np.sum(plan * M)
    return float(distance)


def sinkhorn_barycenter(
    distributions: np.ndarray,
    M_list: list,
    weights: Optional[np.ndarray] = None,
    reg: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Compute Wasserstein barycenter using Sinkhorn algorithm.

    The barycenter minimizes weighted sum of Sinkhorn distances:
        a* = argmin_a sum_i w_i D_reg(a, a_i)

    Parameters
    ----------
    distributions : np.ndarray, shape (k, n)
        k probability distributions of size n
    M_list : list of np.ndarray
        List of k cost matrices, each shape (n, n)
    weights : np.ndarray, shape (k,), optional
        Weights for each distribution (default: uniform)
    reg : float, default=0.01
        Entropic regularization
    max_iter : int, default=100
        Maximum iterations
    tol : float, default=1e-6
        Convergence tolerance
    use_gpu : bool, default=False
        Use GPU acceleration

    Returns
    -------
    barycenter : np.ndarray, shape (n,)
        Wasserstein barycenter distribution

    References
    ----------
    Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters.
    International Conference on Machine Learning.
    """
    k, n = distributions.shape

    if weights is None:
        weights = np.ones(k) / k
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()

    # Initialize barycenter
    barycenter = np.ones(n) / n

    # Iterative Bregman projections
    for iteration in range(max_iter):
        barycenter_prev = barycenter.copy()

        # Compute transport plans to all distributions
        plans = []
        for i in range(k):
            plan = sinkhorn_plan(
                barycenter, distributions[i], M_list[i],
                reg=reg, max_iter=1000, tol=tol, use_gpu=use_gpu
            )
            plans.append(plan)

        # Update barycenter using transport plan marginals
        # Geometric mean of transported distributions
        log_barycenter = np.zeros(n)
        for i, plan in enumerate(plans):
            # Marginal of transport plan
            marginal = plan.sum(axis=1)
            log_barycenter += weights[i] * np.log(marginal + 1e-10)

        barycenter = np.exp(log_barycenter)
        barycenter = barycenter / barycenter.sum()

        # Check convergence
        if np.linalg.norm(barycenter - barycenter_prev) < tol:
            break

    return barycenter


def stabilized_sinkhorn_plan(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Numerically stabilized Sinkhorn algorithm using log-domain computations.

    More stable than standard Sinkhorn for small regularization parameters.

    Parameters
    ----------
    a : np.ndarray, shape (n,)
        Source distribution
    b : np.ndarray, shape (m,)
        Target distribution
    M : np.ndarray, shape (n, m)
        Cost matrix
    reg : float, default=0.01
        Entropic regularization parameter
    max_iter : int, default=1000
        Maximum iterations
    tol : float, default=1e-9
        Convergence tolerance

    Returns
    -------
    plan : np.ndarray, shape (n, m)
        Optimal transport plan

    Notes
    -----
    Uses log-domain arithmetic to avoid numerical overflow/underflow:
        log(u_{k+1}) = log(a) - log_sum_exp(log(K) + log(v_k))

    Essential for small reg or large M values.

    References
    ----------
    Schmitzer, B. (2019). Stabilized sparse scaling algorithms for optimal
    transport. SIAM Journal on Scientific Computing, 41(3), A1443-A1481.
    """
    if not np.allclose(a.sum(), 1.0) or not np.allclose(b.sum(), 1.0):
        warnings.warn("Distributions should sum to 1, normalizing...")
        a = a / a.sum()
        b = b / b.sum()

    n, m = M.shape

    # Log-domain initialization
    log_a = np.log(a + 1e-100)
    log_b = np.log(b + 1e-100)
    K = -M / reg

    # Initialize log-domain dual variables
    log_u = np.zeros(n)
    log_v = np.zeros(m)

    # Sinkhorn iterations in log domain
    for iteration in range(max_iter):
        log_u_prev = log_u.copy()

        # Update log_v
        log_sum_exp_cols = _log_sum_exp(K.T + log_u[None, :], axis=1)
        log_v = log_b - log_sum_exp_cols

        # Update log_u
        log_sum_exp_rows = _log_sum_exp(K + log_v[None, :], axis=1)
        log_u = log_a - log_sum_exp_rows

        # Check convergence
        if iteration % 10 == 0:
            if np.linalg.norm(log_u - log_u_prev) < tol:
                break

    # Compute transport plan in original domain
    log_plan = log_u[:, None] + K + log_v[None, :]
    plan = np.exp(log_plan)

    return plan


def _log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Numerically stable log-sum-exp computation.

    Computes log(sum(exp(x))) in a numerically stable way.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
