"""
Visualization functions for connectivity matrices and their evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, List, Tuple, Union


def plot_connectivity_matrix(
    conn_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Functional Connectivity",
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (8, 7),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a single connectivity matrix as a heatmap.

    Parameters
    ----------
    conn_matrix : np.ndarray, shape (n_regions, n_regions)
        Connectivity matrix to plot
    labels : list of str, optional
        Region labels for axes
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Matplotlib axes to plot on

    Returns
    -------
    ax : plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(conn_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation')

    # Labels
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha='right')
        ax.set_yticklabels(labels)

    ax.set_title(title)
    ax.set_xlabel('Brain Region')
    ax.set_ylabel('Brain Region')

    return ax


def plot_connectivity_evolution(
    conn_matrices: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    n_plots: int = 6,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot evolution of connectivity matrices over time.

    Shows snapshots at evenly spaced time points to visualize network dynamics.

    Parameters
    ----------
    conn_matrices : np.ndarray, shape (n_windows, n_regions, n_regions)
        Time-varying connectivity matrices
    time_points : np.ndarray, optional
        Time values for each window (default: window indices)
    n_plots : int
        Number of snapshots to show
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
    """
    n_windows, n_regions, _ = conn_matrices.shape

    if time_points is None:
        time_points = np.arange(n_windows)

    # Select evenly spaced time points
    indices = np.linspace(0, n_windows - 1, n_plots, dtype=int)

    # Create subplots
    n_rows = (n_plots + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        im = ax.imshow(conn_matrices[idx], cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Time = {time_points[idx]:.1f}')
        ax.set_xlabel('Region')
        ax.set_ylabel('Region')

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Correlation', shrink=0.6)

    fig.suptitle('Connectivity Evolution Over Time', fontsize=16, y=0.995)
    plt.tight_layout()

    return fig


def plot_state_timeline(
    state_labels: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    distances: Optional[np.ndarray] = None,
    state_colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot network state timeline showing transitions.

    Parameters
    ----------
    state_labels : np.ndarray, shape (n_windows,)
        State assignment for each time window
    time_points : np.ndarray, optional
        Time values for each window
    distances : np.ndarray, optional
        Wasserstein distances between windows (plotted below timeline)
    state_colors : list of str, optional
        Colors for each state
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Examples
    --------
    >>> state_labels = segment_network_states(conn_matrices, n_states=3)
    >>> distances = wasserstein_trajectory(conn_matrices)
    >>> fig = plot_state_timeline(state_labels, distances=distances)
    """
    n_windows = len(state_labels)
    n_states = len(np.unique(state_labels))

    if time_points is None:
        time_points = np.arange(n_windows)

    if state_colors is None:
        # Default colors
        cmap = plt.cm.get_cmap('tab10')
        state_colors = [cmap(i) for i in range(n_states)]

    # Create figure
    if distances is not None:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
        ax_timeline = fig.add_subplot(gs[0])
        ax_distance = fig.add_subplot(gs[1], sharex=ax_timeline)
    else:
        fig, ax_timeline = plt.subplots(figsize=figsize)
        ax_distance = None

    # Plot state timeline
    for i in range(n_windows - 1):
        color = state_colors[state_labels[i]]
        ax_timeline.axvspan(
            time_points[i], time_points[i + 1],
            facecolor=color, alpha=0.7, edgecolor='none'
        )

    # Add last segment
    color = state_colors[state_labels[-1]]
    ax_timeline.axvspan(
        time_points[-1], time_points[-1] + (time_points[-1] - time_points[-2]),
        facecolor=color, alpha=0.7, edgecolor='none'
    )

    # Mark transition points
    transitions = np.where(np.diff(state_labels) != 0)[0]
    for trans in transitions:
        ax_timeline.axvline(time_points[trans + 1], color='black', linestyle='--', alpha=0.5)

    ax_timeline.set_ylabel('Network State')
    ax_timeline.set_yticks(range(n_states))
    ax_timeline.set_yticklabels([f'State {i}' for i in range(n_states)])
    ax_timeline.set_title('Network State Timeline')
    ax_timeline.grid(axis='x', alpha=0.3)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=state_colors[i], alpha=0.7, label=f'State {i}')
        for i in range(n_states)
    ]
    ax_timeline.legend(handles=legend_elements, loc='upper right')

    # Plot Wasserstein distance trajectory if provided
    if distances is not None and ax_distance is not None:
        ax_distance.plot(time_points[:-1], distances, 'k-', linewidth=2, alpha=0.7)
        ax_distance.fill_between(
            time_points[:-1], 0, distances, alpha=0.3, color='gray'
        )

        # Highlight transitions
        for trans in transitions:
            ax_distance.axvline(
                time_points[trans + 1], color='red', linestyle='--', alpha=0.5, linewidth=2
            )

        ax_distance.set_xlabel('Time')
        ax_distance.set_ylabel('Wasserstein Distance')
        ax_distance.set_title('Network Reconfiguration (Wasserstein Distance)')
        ax_distance.grid(alpha=0.3)

    return fig


def plot_connectivity_carpet(
    time_series: np.ndarray,
    conn_matrices: Optional[np.ndarray] = None,
    state_labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Create carpet plot showing time series, connectivity, and states together.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        fMRI time series data
    conn_matrices : np.ndarray, optional
        Connectivity matrices to show as lower panel
    state_labels : np.ndarray, optional
        State labels to show as colored bar
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object
    """
    n_timepoints, n_regions = time_series.shape

    # Normalize time series for visualization
    ts_norm = (time_series - time_series.mean(axis=0)) / time_series.std(axis=0)

    # Create subplots
    n_panels = 2 if conn_matrices is not None else 1
    n_panels += 1 if state_labels is not None else 0

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_panels, 1, height_ratios=[3] + [1] * (n_panels - 1), hspace=0.05)

    # Panel 1: Time series carpet plot
    ax_carpet = fig.add_subplot(gs[0])
    im = ax_carpet.imshow(
        ts_norm.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3, interpolation='nearest'
    )
    ax_carpet.set_ylabel('Brain Region')
    ax_carpet.set_title('Time Series Carpet Plot')
    plt.colorbar(im, ax=ax_carpet, label='Normalized Signal')

    panel_idx = 1

    # Panel 2: State labels (if provided)
    if state_labels is not None:
        ax_states = fig.add_subplot(gs[panel_idx], sharex=ax_carpet)

        # Map state labels to colors
        n_states = len(np.unique(state_labels))
        cmap = plt.cm.get_cmap('tab10')
        state_colors = np.array([cmap(s) for s in state_labels])

        # Create image of state labels
        state_image = state_colors[:, :3].T  # RGB only
        ax_states.imshow(
            state_image[None, :, :], aspect='auto', interpolation='nearest'
        )
        ax_states.set_ylabel('State')
        ax_states.set_yticks([])
        ax_states.set_xlim(ax_carpet.get_xlim())

        panel_idx += 1

    # Panel 3: Connectivity evolution (if provided)
    if conn_matrices is not None:
        ax_conn = fig.add_subplot(gs[panel_idx], sharex=ax_carpet)

        # Extract upper triangle and plot as heatmap over time
        n_windows = len(conn_matrices)
        n_edges = (n_regions * (n_regions - 1)) // 2
        conn_edges = np.zeros((n_edges, n_windows))

        for i, conn in enumerate(conn_matrices):
            triu_idx = np.triu_indices(n_regions, k=1)
            conn_edges[:, i] = conn[triu_idx]

        im_conn = ax_conn.imshow(
            conn_edges, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest'
        )
        ax_conn.set_ylabel('Edge')
        ax_conn.set_xlabel('Time Window')
        plt.colorbar(im_conn, ax=ax_conn, label='Correlation')

    plt.tight_layout()

    return fig
