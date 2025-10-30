"""
Dynamic connectivity analysis using optimal transport.

Provides high-level functions for analyzing network evolution,
detecting state transitions, and segmenting time series into discrete states.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy import signal, stats
import warnings


def wasserstein_trajectory(
    conn_matrices: np.ndarray,
    metric: str = "frobenius",
    return_matrices: bool = False
) -> np.ndarray:
    """
    Compute Wasserstein distance trajectory showing network evolution.

    Quantifies how much the network changes between consecutive time windows
    using optimal transport distance.

    Parameters
    ----------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices
    metric : str, default="frobenius"
        Wasserstein metric ('frobenius', 'riemannian', 'bures')
    return_matrices : bool, default=False
        If True, also return pairwise distance matrix

    Returns
    -------
    distances : np.ndarray, shape (n_windows-1,)
        Wasserstein distance between consecutive windows
    distance_matrix : np.ndarray, shape (n_windows, n_windows), optional
        Full pairwise distance matrix (if return_matrices=True)

    Notes
    -----
    Peaks in the distance trajectory indicate network reconfigurations.
    Can be used to detect state transitions.

    Examples
    --------
    >>> from ot_brain_dynamics.connectivity import sliding_window_connectivity
    >>> from ot_brain_dynamics.simulations import generate_var_transitions
    >>> data, states = generate_var_transitions(50, 1000, n_states=3)
    >>> conn = sliding_window_connectivity(data, window_size=50)
    >>> distances = wasserstein_trajectory(conn)
    >>> # Peaks in distances indicate transitions
    """
    from ot_brain_dynamics.optimal_transport import wasserstein_spd

    n_windows = len(conn_matrices)
    distances = np.zeros(n_windows - 1)

    for i in range(n_windows - 1):
        distances[i] = wasserstein_spd(
            conn_matrices[i],
            conn_matrices[i + 1],
            metric=metric
        )

    if return_matrices:
        # Compute full pairwise distance matrix
        distance_matrix = np.zeros((n_windows, n_windows))
        for i in range(n_windows):
            for j in range(i + 1, n_windows):
                dist = wasserstein_spd(
                    conn_matrices[i],
                    conn_matrices[j],
                    metric=metric
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distances, distance_matrix

    return distances


def detect_state_transitions(
    distances: np.ndarray,
    threshold: Optional[float] = None,
    method: str = 'peaks',
    min_separation: int = 5,
    **kwargs
) -> np.ndarray:
    """
    Detect discrete state transitions from Wasserstein distance trajectory.

    Identifies time points where network undergoes significant reconfiguration.

    Parameters
    ----------
    distances : np.ndarray, shape (n_windows-1,)
        Wasserstein distance trajectory from wasserstein_trajectory()
    threshold : float, optional
        Detection threshold (default: mean + 2*std)
    method : str, default='peaks'
        Detection method:
        - 'peaks': Peak detection in distance trajectory
        - 'threshold': Simple threshold crossing
        - 'changepoint': Bayesian changepoint detection
    min_separation : int, default=5
        Minimum time windows between transitions
    **kwargs
        Method-specific parameters

    Returns
    -------
    transition_times : np.ndarray
        Indices of detected state transitions

    Notes
    -----
    The method identifies transitions as peaks or threshold crossings in
    the Wasserstein distance trajectory. High distances indicate the network
    is changing configuration.

    Examples
    --------
    >>> distances = wasserstein_trajectory(conn_matrices)
    >>> transitions = detect_state_transitions(distances, method='peaks')
    >>> print(f"Detected {len(transitions)} transitions at times: {transitions}")
    """
    if threshold is None:
        # Automatic threshold: mean + 2*std
        threshold = distances.mean() + 2 * distances.std()

    if method == 'peaks':
        # Peak detection with prominence threshold
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            distances,
            height=threshold,
            distance=min_separation,
            **kwargs
        )
        return peaks

    elif method == 'threshold':
        # Simple threshold crossing
        above_threshold = distances > threshold

        # Find transitions (rising edges)
        transitions = []
        for i in range(1, len(above_threshold)):
            if above_threshold[i] and not above_threshold[i - 1]:
                transitions.append(i)

        # Enforce minimum separation
        if len(transitions) > 1:
            transitions_filtered = [transitions[0]]
            for t in transitions[1:]:
                if t - transitions_filtered[-1] >= min_separation:
                    transitions_filtered.append(t)
            transitions = transitions_filtered

        return np.array(transitions)

    elif method == 'changepoint':
        # Bayesian changepoint detection
        return _bayesian_changepoint_detection(distances, threshold, min_separation)

    else:
        raise ValueError(f"Unknown method: {method}")


def _bayesian_changepoint_detection(
    signal: np.ndarray,
    threshold: float,
    min_separation: int
) -> np.ndarray:
    """
    Bayesian changepoint detection using cumulative sum approach.

    Detects points where the signal mean changes significantly.
    """
    # Compute cumulative sum after removing mean
    signal_centered = signal - signal.mean()
    cumsum = np.cumsum(signal_centered)

    # Detect changepoints as peaks in |cumsum|
    from scipy.signal import find_peaks

    peaks_pos, _ = find_peaks(cumsum, distance=min_separation)
    peaks_neg, _ = find_peaks(-cumsum, distance=min_separation)

    # Combine and sort
    changepoints = np.sort(np.concatenate([peaks_pos, peaks_neg]))

    # Filter by significance (using local variance)
    significant_changepoints = []
    window = min_separation * 2

    for cp in changepoints:
        start = max(0, cp - window)
        end = min(len(signal), cp + window)

        before = signal[start:cp]
        after = signal[cp:end]

        if len(before) > 2 and len(after) > 2:
            # T-test for mean difference
            _, p_value = stats.ttest_ind(before, after)
            if p_value < 0.05:  # Significant difference
                significant_changepoints.append(cp)

    return np.array(significant_changepoints)


def segment_network_states(
    conn_matrices: np.ndarray,
    n_states: Optional[int] = None,
    method: str = 'kmeans',
    metric: str = 'frobenius',
    return_centers: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Segment connectivity time series into discrete network states.

    Clusters connectivity matrices to identify recurring network configurations.

    Parameters
    ----------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices
    n_states : int, optional
        Number of states to identify (default: auto-select using elbow method)
    method : str, default='kmeans'
        Clustering method:
        - 'kmeans': K-means on Wasserstein manifold
        - 'hierarchical': Hierarchical clustering
        - 'hmm': Hidden Markov model
    metric : str, default='frobenius'
        Distance metric for clustering
    return_centers : bool, default=False
        If True, also return state centroid matrices
    **kwargs
        Method-specific parameters

    Returns
    -------
    state_labels : np.ndarray, shape (n_windows,)
        State assignment for each time window
    state_centers : np.ndarray, shape (n_states, n_regions, n_regions), optional
        State centroid connectivity matrices (if return_centers=True)

    Notes
    -----
    This performs clustering on the SPD manifold of connectivity matrices
    using Wasserstein distance. State centers are Wasserstein barycenters.

    Examples
    --------
    >>> conn_matrices = sliding_window_connectivity(time_series)
    >>> state_labels, centers = segment_network_states(
    ...     conn_matrices, n_states=3, return_centers=True
    ... )
    >>> # Analyze state statistics
    >>> stats = compute_state_statistics(state_labels)
    """
    from ot_brain_dynamics.optimal_transport import wasserstein_spd, wasserstein_barycenter_spd

    n_windows, n_regions, _ = conn_matrices.shape

    # Auto-select number of states if not provided
    if n_states is None:
        n_states = _estimate_n_states(conn_matrices, metric)

    if method == 'kmeans':
        # K-means clustering on SPD manifold
        state_labels, state_centers = _kmeans_spd(
            conn_matrices, n_states, metric, **kwargs
        )

    elif method == 'hierarchical':
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from ot_brain_dynamics.optimal_transport import wasserstein_distance_matrix

        # Compute distance matrix
        dist_matrix = wasserstein_distance_matrix(conn_matrices, metric=metric)

        # Hierarchical clustering
        linkage_matrix = linkage(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        state_labels = fcluster(linkage_matrix, n_states, criterion='maxclust') - 1

        # Compute state centers if requested
        if return_centers:
            state_centers = _compute_state_centers(conn_matrices, state_labels, n_states, metric)
    else:
        raise ValueError(f"Unknown method: {method}")

    if return_centers:
        return state_labels, state_centers
    else:
        return state_labels


def _kmeans_spd(
    matrices: np.ndarray,
    n_states: int,
    metric: str,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering on SPD manifold."""
    from ot_brain_dynamics.optimal_transport import wasserstein_spd, wasserstein_barycenter_spd

    n_windows = len(matrices)
    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for init_idx in range(n_init):
        # Random initialization
        init_indices = np.random.choice(n_windows, size=n_states, replace=False)
        centers = matrices[init_indices].copy()

        for iteration in range(max_iter):
            # Assignment step
            labels = np.zeros(n_windows, dtype=int)
            for i, matrix in enumerate(matrices):
                distances = np.array([
                    wasserstein_spd(matrix, center, metric=metric)
                    for center in centers
                ])
                labels[i] = np.argmin(distances)

            # Update step - compute Wasserstein barycenter for each cluster
            centers_new = np.zeros_like(centers)
            for k in range(n_states):
                cluster_matrices = matrices[labels == k]
                if len(cluster_matrices) > 0:
                    if metric == 'frobenius':
                        # Closed form for Frobenius metric
                        barycenter, _ = wasserstein_barycenter_spd(
                            cluster_matrices, metric='frobenius'
                        )
                    else:
                        barycenter, _ = wasserstein_barycenter_spd(
                            cluster_matrices, metric=metric, max_iter=50
                        )
                    centers_new[k] = barycenter
                else:
                    centers_new[k] = centers[k]  # Keep old center

            # Check convergence
            center_change = sum([
                wasserstein_spd(centers[k], centers_new[k], metric=metric)
                for k in range(n_states)
            ])

            centers = centers_new

            if center_change < tol:
                break

        # Compute inertia (total within-cluster distance)
        inertia = 0.0
        for i, matrix in enumerate(matrices):
            inertia += wasserstein_spd(matrix, centers[labels[i]], metric=metric) ** 2

        # Keep best result
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers


def _estimate_n_states(matrices: np.ndarray, metric: str) -> int:
    """Estimate optimal number of states using elbow method."""
    from ot_brain_dynamics.optimal_transport import wasserstein_distance_matrix

    max_states = min(10, len(matrices) // 10)
    inertias = []

    for k in range(2, max_states + 1):
        labels, centers = _kmeans_spd(matrices, k, metric, max_iter=50, n_init=3)

        # Compute inertia
        from ot_brain_dynamics.optimal_transport import wasserstein_spd
        inertia = sum([
            wasserstein_spd(matrices[i], centers[labels[i]], metric=metric) ** 2
            for i in range(len(matrices))
        ])
        inertias.append(inertia)

    # Find elbow using second derivative
    inertias = np.array(inertias)
    if len(inertias) > 2:
        second_deriv = np.diff(inertias, n=2)
        elbow_idx = np.argmax(second_deriv) + 2  # +2 because we start at k=2
        return elbow_idx
    else:
        return 3  # Default


def _compute_state_centers(
    matrices: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    metric: str
) -> np.ndarray:
    """Compute state center matrices as barycenters."""
    from ot_brain_dynamics.optimal_transport import wasserstein_barycenter_spd

    centers = []
    for k in range(n_states):
        cluster_matrices = matrices[labels == k]
        if len(cluster_matrices) > 0:
            barycenter, _ = wasserstein_barycenter_spd(cluster_matrices, metric=metric)
            centers.append(barycenter)
        else:
            # Empty cluster - use random matrix
            centers.append(matrices[np.random.randint(len(matrices))])

    return np.array(centers)


def compute_state_statistics(
    state_labels: np.ndarray,
    distances: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute statistics about network state dynamics.

    Parameters
    ----------
    state_labels : np.ndarray, shape (n_windows,)
        State assignments from segment_network_states()
    distances : np.ndarray, shape (n_windows-1,), optional
        Wasserstein distances from wasserstein_trajectory()

    Returns
    -------
    statistics : dict
        Dictionary containing:
        - 'occupancy': Proportion of time in each state
        - 'dwell_time': Average consecutive time in each state
        - 'transition_matrix': State transition probability matrix
        - 'transition_distances': Average distance for each transition type

    Notes
    -----
    These statistics characterize:
    - Which states are most common (occupancy)
    - How stable each state is (dwell time)
    - Which state transitions are likely (transition matrix)
    - How abrupt transitions are (transition distances)

    Examples
    --------
    >>> state_labels = segment_network_states(conn_matrices, n_states=3)
    >>> stats = compute_state_statistics(state_labels)
    >>> print(f"State occupancies: {stats['occupancy']}")
    >>> print(f"Transition matrix:\\n{stats['transition_matrix']}")
    """
    n_windows = len(state_labels)
    n_states = len(np.unique(state_labels))

    # Occupancy
    occupancy = np.array([
        (state_labels == k).sum() / n_windows
        for k in range(n_states)
    ])

    # Dwell time (consecutive windows in same state)
    dwell_times = {k: [] for k in range(n_states)}

    current_state = state_labels[0]
    current_dwell = 1

    for i in range(1, n_windows):
        if state_labels[i] == current_state:
            current_dwell += 1
        else:
            dwell_times[current_state].append(current_dwell)
            current_state = state_labels[i]
            current_dwell = 1

    dwell_times[current_state].append(current_dwell)

    # Average dwell time per state
    avg_dwell_time = np.array([
        np.mean(dwell_times[k]) if len(dwell_times[k]) > 0 else 0
        for k in range(n_states)
    ])

    # Transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_windows - 1):
        from_state = state_labels[i]
        to_state = state_labels[i + 1]
        transition_matrix[from_state, to_state] += 1

    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums

    statistics = {
        'occupancy': occupancy,
        'dwell_time': avg_dwell_time,
        'transition_matrix': transition_matrix,
    }

    # Transition distances if provided
    if distances is not None:
        transition_distances = np.zeros((n_states, n_states))
        transition_counts = np.zeros((n_states, n_states))

        for i in range(n_windows - 1):
            from_state = state_labels[i]
            to_state = state_labels[i + 1]
            transition_distances[from_state, to_state] += distances[i]
            transition_counts[from_state, to_state] += 1

        # Average distance per transition type
        transition_counts[transition_counts == 0] = 1  # Avoid division by zero
        transition_distances = transition_distances / transition_counts

        statistics['transition_distances'] = transition_distances

    return statistics
