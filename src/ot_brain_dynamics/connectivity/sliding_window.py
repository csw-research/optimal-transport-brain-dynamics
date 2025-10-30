"""
Sliding window approaches for dynamic connectivity estimation.

Implements various windowing strategies for computing time-resolved
functional connectivity from fMRI time series.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import warnings


def sliding_window_connectivity(
    time_series: np.ndarray,
    window_size: int = 50,
    step_size: Optional[int] = None,
    method: str = 'correlation',
    tapered: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Compute sliding window functional connectivity matrices.

    Estimates time-varying connectivity by computing correlation/covariance
    matrices within overlapping temporal windows.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        fMRI time series data (each column is a brain region)
    window_size : int, default=50
        Size of sliding window in time points
    step_size : int, optional
        Step between windows (default: window_size // 2 for 50% overlap)
    method : str, default='correlation'
        Connectivity metric:
        - 'correlation': Pearson correlation
        - 'covariance': Covariance matrix
        - 'partial': Partial correlation
        - 'precision': Precision matrix (inverse covariance)
    tapered : bool, default=True
        Apply Hamming window tapering to reduce edge effects
    **kwargs
        Additional arguments for covariance estimation (e.g., regularization)

    Returns
    -------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices

    Notes
    -----
    Window size selection is critical:
    - Too small: noisy estimates, poor covariance conditioning
    - Too large: over-smooths rapid transitions

    Rule of thumb: window_size >= 3 * n_regions for stable estimates

    Tapering reduces spurious correlations from window edges but slightly
    reduces effective window size.

    Examples
    --------
    >>> from ot_brain_dynamics.simulations import generate_var_transitions
    >>> data, states = generate_var_transitions(50, 1000)
    >>> conn = sliding_window_connectivity(data, window_size=50)
    >>> print(conn.shape)  # (n_windows, 50, 50)
    """
    n_timepoints, n_regions = time_series.shape

    if window_size >= n_timepoints:
        raise ValueError(
            f"window_size ({window_size}) must be less than "
            f"n_timepoints ({n_timepoints})"
        )

    if window_size < n_regions:
        warnings.warn(
            f"window_size ({window_size}) < n_regions ({n_regions}). "
            "Covariance estimates may be poorly conditioned. "
            "Consider increasing window_size or using regularization."
        )

    if step_size is None:
        step_size = window_size // 2  # 50% overlap by default

    # Generate window tapering if requested
    if tapered:
        taper = np.hamming(window_size)
    else:
        taper = np.ones(window_size)

    # Compute number of windows
    n_windows = (n_timepoints - window_size) // step_size + 1

    # Initialize output
    conn_matrices = np.zeros((n_windows, n_regions, n_regions))

    # Slide window
    for w_idx in range(n_windows):
        start = w_idx * step_size
        end = start + window_size

        if end > n_timepoints:
            break

        # Extract window
        window_data = time_series[start:end, :]

        # Apply tapering
        if tapered:
            window_data = window_data * taper[:, None]

        # Compute connectivity
        conn_matrices[w_idx] = _compute_connectivity(window_data, method, **kwargs)

    return conn_matrices[:w_idx + 1]  # Trim unused windows


def _compute_connectivity(
    data: np.ndarray,
    method: str,
    **kwargs
) -> np.ndarray:
    """Compute connectivity matrix from windowed data."""
    from ot_brain_dynamics.connectivity.covariance_estimation import (
        robust_covariance_estimate,
        shrinkage_covariance,
    )

    n_timepoints, n_regions = data.shape

    if method == 'correlation':
        # Standardize data
        data_centered = data - data.mean(axis=0, keepdims=True)
        data_std = data.std(axis=0, keepdims=True)
        data_std[data_std == 0] = 1.0  # Avoid division by zero
        data_norm = data_centered / data_std

        # Compute correlation
        corr = (data_norm.T @ data_norm) / (n_timepoints - 1)

        # Ensure valid correlation matrix
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2  # Ensure symmetry

        return corr

    elif method == 'covariance':
        # Use shrinkage by default for better conditioning
        cov = shrinkage_covariance(data, **kwargs)
        return cov

    elif method == 'partial':
        # Partial correlation via precision matrix
        cov = shrinkage_covariance(data, **kwargs)
        try:
            prec = np.linalg.inv(cov)
            # Convert precision to partial correlation
            D = np.sqrt(np.diag(prec))
            partial_corr = -prec / np.outer(D, D)
            np.fill_diagonal(partial_corr, 1.0)
            return partial_corr
        except np.linalg.LinAlgError:
            warnings.warn("Precision matrix singular, returning correlation instead")
            return _compute_connectivity(data, 'correlation')

    elif method == 'precision':
        cov = shrinkage_covariance(data, **kwargs)
        try:
            prec = np.linalg.inv(cov)
            return prec
        except np.linalg.LinAlgError:
            warnings.warn("Precision matrix singular, returning pseudo-inverse")
            return np.linalg.pinv(cov)

    else:
        raise ValueError(f"Unknown method: {method}")


def adaptive_window_connectivity(
    time_series: np.ndarray,
    min_window: int = 30,
    max_window: int = 100,
    alpha: float = 0.05,
    method: str = 'correlation',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute connectivity with adaptive window sizes based on signal stability.

    Automatically adjusts window size to balance temporal resolution
    and estimation accuracy based on local signal characteristics.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        fMRI time series data
    min_window : int, default=30
        Minimum window size
    max_window : int, default=100
        Maximum window size
    alpha : float, default=0.05
        Sensitivity parameter (lower = larger windows for more stable estimates)
    method : str, default='correlation'
        Connectivity metric

    Returns
    -------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices
    window_sizes : np.ndarray, shape (n_windows,)
        Selected window size for each connectivity matrix

    Notes
    -----
    Adaptive windowing:
    - Uses larger windows when signal is stable (low variance)
    - Uses smaller windows when signal changes rapidly (high variance)
    - Provides better temporal resolution during state transitions

    The algorithm estimates local signal variance and adjusts window size
    inversely proportional to variance.

    References
    ----------
    Leonardi, N., & Van De Ville, D. (2015). On spurious and real fluctuations
    of dynamic functional connectivity during rest. Neuroimage, 104, 430-436.
    """
    n_timepoints, n_regions = time_series.shape

    # Compute local variance using sliding window
    variance_window = min_window
    local_variance = np.zeros(n_timepoints - variance_window + 1)

    for i in range(len(local_variance)):
        window_data = time_series[i:i + variance_window]
        local_variance[i] = np.var(window_data)

    # Smooth variance estimate
    from scipy.ndimage import gaussian_filter1d
    local_variance = gaussian_filter1d(local_variance, sigma=5)

    # Determine adaptive window sizes
    # High variance -> small windows, low variance -> large windows
    variance_norm = (local_variance - local_variance.min()) / (
        local_variance.max() - local_variance.min() + 1e-10
    )

    window_sizes_float = max_window - (max_window - min_window) * variance_norm
    window_sizes = window_sizes_float.astype(int)

    # Compute connectivity with adaptive windows
    conn_matrices = []
    final_window_sizes = []

    i = 0
    while i < n_timepoints:
        # Get window size for current position
        variance_idx = min(i, len(window_sizes) - 1)
        window_size = window_sizes[variance_idx]

        # Ensure we have enough data
        if i + window_size > n_timepoints:
            window_size = n_timepoints - i

        if window_size < min_window:
            break

        # Extract window
        window_data = time_series[i:i + window_size]

        # Compute connectivity
        conn = _compute_connectivity(window_data, method)
        conn_matrices.append(conn)
        final_window_sizes.append(window_size)

        # Move forward (50% overlap)
        i += window_size // 2

    return np.array(conn_matrices), np.array(final_window_sizes)


def estimate_optimal_window_size(
    time_series: np.ndarray,
    window_range: Tuple[int, int] = (20, 100),
    metric: str = 'stability',
    n_repeats: int = 10,
    random_state: Optional[int] = None
) -> Tuple[int, dict]:
    """
    Estimate optimal window size for sliding window connectivity analysis.

    Uses cross-validation to select window size that balances temporal
    resolution and estimation reliability.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        fMRI time series data
    window_range : tuple of int, default=(20, 100)
        Range of window sizes to test (min, max)
    metric : str, default='stability'
        Optimization criterion:
        - 'stability': Minimize variance across adjacent windows
        - 'information': Maximize information content (condition number)
        - 'cross_val': Cross-validation correlation
    n_repeats : int, default=10
        Number of bootstrap repeats for stability estimation
    random_state : int, optional
        Random seed

    Returns
    -------
    optimal_window : int
        Recommended window size
    results : dict
        Dictionary with 'window_sizes' and 'scores' for all tested sizes

    Notes
    -----
    The stability metric looks for window sizes where connectivity estimates
    are reliable (low noise) but still capture temporal dynamics (not over-smoothed).

    Typically, optimal window ~50-80 timepoints for resting-state fMRI with TR~2s.

    Examples
    --------
    >>> from ot_brain_dynamics.simulations import generate_var_transitions
    >>> data, _ = generate_var_transitions(50, 1000)
    >>> optimal_w, results = estimate_optimal_window_size(data)
    >>> print(f"Optimal window size: {optimal_w}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    min_w, max_w = window_range
    window_sizes = np.arange(min_w, max_w + 1, 5)
    scores = np.zeros(len(window_sizes))

    for i, window_size in enumerate(window_sizes):
        if metric == 'stability':
            score = _evaluate_stability(time_series, window_size, n_repeats)
        elif metric == 'information':
            score = _evaluate_information(time_series, window_size)
        elif metric == 'cross_val':
            score = _evaluate_cross_validation(time_series, window_size)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores[i] = score

    # Select window with best score
    # For stability and cross_val, higher is better
    # For information, we want balanced (not too high or low)
    if metric in ['stability', 'cross_val']:
        optimal_idx = np.argmax(scores)
    else:
        optimal_idx = np.argmax(scores)

    optimal_window = int(window_sizes[optimal_idx])

    results = {
        'window_sizes': window_sizes,
        'scores': scores,
        'metric': metric
    }

    return optimal_window, results


def _evaluate_stability(
    time_series: np.ndarray,
    window_size: int,
    n_repeats: int
) -> float:
    """Evaluate stability of connectivity estimates using bootstrap."""
    n_timepoints, n_regions = time_series.shape

    # Compute connectivity for several random window placements
    conn_estimates = []

    for _ in range(n_repeats):
        # Random window placement
        start = np.random.randint(0, n_timepoints - window_size)
        window_data = time_series[start:start + window_size]

        # Bootstrap sample within window
        boot_idx = np.random.choice(window_size, size=window_size, replace=True)
        boot_data = window_data[boot_idx]

        conn = _compute_connectivity(boot_data, 'correlation')
        conn_estimates.append(conn)

    # Compute variance across estimates (lower = more stable)
    conn_stack = np.array(conn_estimates)
    variance = np.var(conn_stack, axis=0).mean()

    # Return negative variance (so higher score = better)
    return -variance


def _evaluate_information(time_series: np.ndarray, window_size: int) -> float:
    """Evaluate information content using condition number."""
    n_timepoints = len(time_series)

    # Sample a window
    start = (n_timepoints - window_size) // 2
    window_data = time_series[start:start + window_size]

    # Compute covariance
    cov = _compute_connectivity(window_data, 'covariance')

    # Condition number (lower = better conditioned)
    # We want moderate conditioning - not too ill-conditioned but not trivial
    eigvals = np.linalg.eigvalsh(cov)
    condition_number = eigvals.max() / (eigvals.min() + 1e-10)

    # Score inversely related to condition number
    score = 1.0 / (1.0 + np.log10(condition_number))

    return score


def _evaluate_cross_validation(time_series: np.ndarray, window_size: int) -> float:
    """Evaluate using cross-validation correlation."""
    n_timepoints = len(time_series)

    # Split into two halves
    mid = n_timepoints // 2

    # Compute connectivity on both halves
    if window_size <= mid and window_size <= (n_timepoints - mid):
        data1 = time_series[:mid]
        data2 = time_series[mid:]

        # Sample windows from each half
        start1 = (mid - window_size) // 2
        start2 = (n_timepoints - mid - window_size) // 2

        conn1 = _compute_connectivity(data1[start1:start1 + window_size], 'correlation')
        conn2 = _compute_connectivity(data2[start2:start2 + window_size], 'correlation')

        # Correlation between upper triangles
        triu_idx = np.triu_indices_from(conn1, k=1)
        corr = np.corrcoef(conn1[triu_idx], conn2[triu_idx])[0, 1]

        return corr if not np.isnan(corr) else 0.0
    else:
        return 0.0
