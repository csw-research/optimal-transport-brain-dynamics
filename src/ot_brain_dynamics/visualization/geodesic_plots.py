"""
Visualization functions for geodesics and Wasserstein distances.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, List, Tuple


def plot_geodesic_path(
    C1: np.ndarray,
    C2: np.ndarray,
    n_steps: int = 10,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Plot geodesic path between two connectivity matrices on SPD manifold.

    Shows interpolation between network states along the unique geodesic.

    Parameters
    ----------
    C1 : np.ndarray, shape (n_regions, n_regions)
        Starting connectivity matrix
    C2 : np.ndarray, shape (n_regions, n_regions)
        Ending connectivity matrix
    n_steps : int
        Number of intermediate points to show
    cmap : str
        Colormap
    vmin, vmax : float
        Color scale limits
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Examples
    --------
    >>> # Show transition between two network states
    >>> from ot_brain_dynamics.simulations import generate_correlation_states
    >>> states = generate_correlation_states(2, 50)
    >>> fig = plot_geodesic_path(states[0], states[1], n_steps=8)
    """
    from ot_brain_dynamics.optimal_transport import geodesic_spd

    # Generate geodesic interpolation
    t_params = np.linspace(0, 1, n_steps)
    geodesic_matrices = geodesic_spd(C1, C2, t_params)

    # Create subplots
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)

    for i, (t, C_t) in enumerate(zip(t_params, geodesic_matrices)):
        ax = axes[i]
        im = ax.imshow(C_t, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f't = {t:.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

        # Label start and end
        if i == 0:
            ax.set_ylabel('Start', fontsize=12, fontweight='bold')
        elif i == n_steps - 1:
            ax.set_ylabel('End', fontsize=12, fontweight='bold')

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Correlation', shrink=0.8)

    fig.suptitle('Geodesic Path Between Network States', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_wasserstein_trajectory(
    distances: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    transitions: Optional[np.ndarray] = None,
    smoothing: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot Wasserstein distance trajectory showing network evolution.

    Parameters
    ----------
    distances : np.ndarray, shape (n_windows-1,)
        Wasserstein distances between consecutive windows
    time_points : np.ndarray, optional
        Time values for each window
    transitions : np.ndarray, optional
        Detected transition times to mark with vertical lines
    smoothing : int, optional
        Window size for smoothing trajectory
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Examples
    --------
    >>> from ot_brain_dynamics.connectivity import (
    ...     sliding_window_connectivity, wasserstein_trajectory, detect_state_transitions
    ... )
    >>> conn = sliding_window_connectivity(time_series)
    >>> distances = wasserstein_trajectory(conn)
    >>> transitions = detect_state_transitions(distances)
    >>> fig = plot_wasserstein_trajectory(distances, transitions=transitions)
    """
    n_windows = len(distances)

    if time_points is None:
        time_points = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=figsize)

    # Apply smoothing if requested
    if smoothing is not None:
        from scipy.ndimage import uniform_filter1d
        distances_smooth = uniform_filter1d(distances, size=smoothing)
        ax.plot(time_points, distances_smooth, 'b-', linewidth=2, label='Smoothed', alpha=0.7)
        ax.plot(time_points, distances, 'k-', linewidth=1, alpha=0.3, label='Raw')
    else:
        ax.plot(time_points, distances, 'b-', linewidth=2, alpha=0.7)

    # Fill area under curve
    ax.fill_between(time_points, 0, distances, alpha=0.2, color='blue')

    # Mark detected transitions
    if transitions is not None:
        for trans in transitions:
            ax.axvline(time_points[trans], color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.text(
                time_points[trans], ax.get_ylim()[1] * 0.9,
                'Transition', rotation=90, va='top', ha='right', fontsize=10
            )

    # Add threshold line (mean + 2*std)
    threshold = distances.mean() + 2 * distances.std()
    ax.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='Threshold (μ + 2σ)')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Wasserstein Distance', fontsize=12)
    ax.set_title('Network Evolution: Wasserstein Distance Trajectory', fontsize=14)
    ax.grid(alpha=0.3)

    if smoothing is not None or transitions is not None:
        ax.legend()

    plt.tight_layout()

    return fig


def plot_distance_matrix(
    distance_matrix: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    state_labels: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot pairwise Wasserstein distance matrix between time windows.

    Reveals temporal structure and recurring states in network dynamics.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n_windows, n_windows)
        Pairwise distance matrix from wasserstein_trajectory()
    time_points : np.ndarray, optional
        Time values for each window
    state_labels : np.ndarray, optional
        State assignments to overlay on plot
    cmap : str
        Colormap
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Notes
    -----
    Block structure in the distance matrix indicates recurring states.
    Low distances (dark colors) show similar network configurations.

    Examples
    --------
    >>> _, dist_matrix = wasserstein_trajectory(conn_matrices, return_matrices=True)
    >>> state_labels = segment_network_states(conn_matrices, n_states=3)
    >>> fig = plot_distance_matrix(dist_matrix, state_labels=state_labels)
    """
    n_windows = len(distance_matrix)

    if time_points is None:
        time_points = np.arange(n_windows)

    fig = plt.figure(figsize=figsize)

    if state_labels is not None:
        gs = GridSpec(2, 2, height_ratios=[1, 10], width_ratios=[10, 1], hspace=0.02, wspace=0.02)
        ax_state_h = fig.add_subplot(gs[0, 0])
        ax_matrix = fig.add_subplot(gs[1, 0], sharex=ax_state_h)
        ax_state_v = fig.add_subplot(gs[1, 1], sharey=ax_matrix)
    else:
        ax_matrix = fig.add_subplot(111)

    # Plot distance matrix
    im = ax_matrix.imshow(distance_matrix, cmap=cmap, aspect='auto', origin='lower')
    ax_matrix.set_xlabel('Time Window', fontsize=12)
    ax_matrix.set_ylabel('Time Window', fontsize=12)
    ax_matrix.set_title('Pairwise Wasserstein Distance Matrix', fontsize=14)

    # Colorbar
    plt.colorbar(im, ax=ax_matrix, label='Wasserstein Distance')

    # Overlay state labels if provided
    if state_labels is not None:
        n_states = len(np.unique(state_labels))
        cmap_states = plt.cm.get_cmap('tab10')
        state_colors = np.array([cmap_states(s) for s in state_labels])

        # Horizontal state bar
        ax_state_h.imshow(
            state_colors[:, :3][None, :, :], aspect='auto', interpolation='nearest'
        )
        ax_state_h.set_xticks([])
        ax_state_h.set_yticks([])
        ax_state_h.set_ylabel('State', fontsize=10)

        # Vertical state bar
        ax_state_v.imshow(
            state_colors[:, :3][:, None, :], aspect='auto', interpolation='nearest'
        )
        ax_state_v.set_xticks([])
        ax_state_v.set_yticks([])
        ax_state_v.set_xlabel('State', fontsize=10)

    plt.tight_layout()

    return fig


def plot_manifold_embedding(
    conn_matrices: np.ndarray,
    state_labels: Optional[np.ndarray] = None,
    method: str = 'mds',
    metric: str = 'frobenius',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Embed connectivity matrices in 2D/3D using manifold learning.

    Visualizes the geometry of network state space.

    Parameters
    ----------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices
    state_labels : np.ndarray, optional
        State assignments for coloring
    method : str
        Embedding method ('mds', 'tsne', 'isomap')
    metric : str
        Wasserstein metric for computing distances
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Notes
    -----
    The embedding reveals the intrinsic dimensionality and structure
    of network state space. Clusters indicate recurring states.

    Examples
    --------
    >>> conn_matrices = sliding_window_connectivity(time_series)
    >>> state_labels = segment_network_states(conn_matrices)
    >>> fig = plot_manifold_embedding(conn_matrices, state_labels, method='mds')
    """
    from sklearn.manifold import MDS, TSNE, Isomap
    from ot_brain_dynamics.optimal_transport import wasserstein_distance_matrix

    # Compute distance matrix
    dist_matrix = wasserstein_distance_matrix(conn_matrices, metric=metric)

    # Compute embedding
    if method == 'mds':
        embedder = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embedding = embedder.fit_transform(dist_matrix)
    elif method == 'tsne':
        embedder = TSNE(n_components=2, metric='precomputed', random_state=42)
        embedding = embedder.fit_transform(dist_matrix)
    elif method == 'isomap':
        embedder = Isomap(n_components=2, metric='precomputed')
        embedding = embedder.fit_transform(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot embedding
    fig, ax = plt.subplots(figsize=figsize)

    if state_labels is not None:
        n_states = len(np.unique(state_labels))
        cmap = plt.cm.get_cmap('tab10')

        for state in range(n_states):
            mask = state_labels == state
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[cmap(state)], s=50, alpha=0.7, label=f'State {state}'
            )

        # Draw trajectory
        ax.plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.2, linewidth=1)

        # Mark start and end
        ax.scatter(embedding[0, 0], embedding[0, 1], c='green', s=200, marker='*',
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(embedding[-1, 0], embedding[-1, 1], c='red', s=200, marker='*',
                  edgecolors='black', linewidths=2, label='End', zorder=5)

        ax.legend()
    else:
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(embedding)))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=50, alpha=0.7)
        ax.plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.2, linewidth=1)

        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(embedding)))
        plt.colorbar(sm, ax=ax, label='Time Window')

    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    ax.set_title(f'Network State Space Embedding ({method.upper()})', fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
