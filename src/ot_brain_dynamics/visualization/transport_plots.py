"""
Visualization functions for optimal transport plans and network reorganization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Optional, Tuple, List


def plot_transport_plan(
    transport_plan: np.ndarray,
    C1: Optional[np.ndarray] = None,
    C2: Optional[np.ndarray] = None,
    threshold: float = 0.01,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualize optimal transport plan showing network reorganization.

    Parameters
    ----------
    transport_plan : np.ndarray, shape (n, n)
        Optimal transport plan matrix
    C1 : np.ndarray, optional
        Source connectivity matrix (shown on left)
    C2 : np.ndarray, optional
        Target connectivity matrix (shown on right)
    threshold : float
        Minimum transport value to display
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Notes
    -----
    The transport plan shows how "mass" (network structure) moves from
    source to target configuration. Strong connections in the plan indicate
    which source features map to which target features.

    Examples
    --------
    >>> from ot_brain_dynamics.optimal_transport import optimal_transport_plan
    >>> plan = optimal_transport_plan(state1, state2)
    >>> fig = plot_transport_plan(plan, state1, state2)
    """
    if C1 is not None and C2 is not None:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax_plan = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        # Plot source connectivity
        im1 = ax1.imshow(C1, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_title('Source State')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Region')
        plt.colorbar(im1, ax=ax1, label='Correlation')

        # Plot target connectivity
        im2 = ax2.imshow(C2, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax2.set_title('Target State')
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Region')
        plt.colorbar(im2, ax=ax2, label='Correlation')

    else:
        fig, ax_plan = plt.subplots(figsize=(8, 7))

    # Plot transport plan
    # Threshold for visualization
    plan_viz = transport_plan.copy()
    plan_viz[plan_viz < threshold] = 0

    im_plan = ax_plan.imshow(plan_viz, cmap='YlOrRd', aspect='auto', origin='lower')
    ax_plan.set_title('Optimal Transport Plan', fontsize=14)
    ax_plan.set_xlabel('Target Feature', fontsize=12)
    ax_plan.set_ylabel('Source Feature', fontsize=12)
    plt.colorbar(im_plan, ax=ax_plan, label='Transport Amount')

    # Add grid for clarity
    ax_plan.grid(False)

    plt.tight_layout()

    return fig


def plot_network_reorganization(
    C1: np.ndarray,
    C2: np.ndarray,
    transport_plan: Optional[np.ndarray] = None,
    n_top_connections: int = 10,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualize how network connections reorganize between states.

    Shows strongest connections in source and target, with transport plan
    indicating the reorganization.

    Parameters
    ----------
    C1 : np.ndarray, shape (n_regions, n_regions)
        Source connectivity matrix
    C2 : np.ndarray, shape (n_regions, n_regions)
        Target connectivity matrix
    transport_plan : np.ndarray, optional
        Optimal transport plan (if not provided, computed automatically)
    n_top_connections : int
        Number of top connections to highlight
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Notes
    -----
    This visualization helps interpret:
    - Which connections strengthen/weaken
    - Which regions become more/less central
    - How modular structure changes
    """
    if transport_plan is None:
        from ot_brain_dynamics.optimal_transport import optimal_transport_plan
        transport_plan = optimal_transport_plan(C1, C2)

    n_regions = C1.shape[0]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Top row: connectivity matrices
    ax1 = fig.add_subplot(gs[0, 0])
    ax_diff = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Plot source
    im1 = ax1.imshow(C1, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('Source Network', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Correlation')

    # Plot difference
    diff = C2 - C1
    vmax_diff = max(abs(diff.min()), abs(diff.max()))
    im_diff = ax_diff.imshow(diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    ax_diff.set_title('Change (Target - Source)', fontweight='bold')
    plt.colorbar(im_diff, ax=ax_diff, label='Δ Correlation')

    # Plot target
    im2 = ax2.imshow(C2, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title('Target Network', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Correlation')

    # Bottom row: connection strength changes
    ax_bar = fig.add_subplot(gs[1, :])

    # Compute connection strength change
    # Use upper triangle (unique connections)
    triu_idx = np.triu_indices(n_regions, k=1)
    conn1_vec = C1[triu_idx]
    conn2_vec = C2[triu_idx]
    conn_change = np.abs(conn2_vec) - np.abs(conn1_vec)

    # Sort by magnitude of change
    sorted_idx = np.argsort(np.abs(conn_change))[::-1]

    # Plot top changes
    top_idx = sorted_idx[:n_top_connections]
    changes_to_plot = conn_change[top_idx]
    colors = ['red' if c > 0 else 'blue' for c in changes_to_plot]

    ax_bar.barh(range(n_top_connections), changes_to_plot, color=colors, alpha=0.7)
    ax_bar.set_yticks(range(n_top_connections))

    # Create labels showing which connections
    labels = []
    for idx in top_idx:
        # Convert back to matrix indices
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        labels.append(f'R{i}-R{j}')

    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel('Change in |Correlation|', fontsize=11)
    ax_bar.set_title(f'Top {n_top_connections} Connection Changes', fontsize=12)
    ax_bar.axvline(0, color='black', linewidth=1)
    ax_bar.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Strengthening'),
        Patch(facecolor='blue', alpha=0.7, label='Weakening')
    ]
    ax_bar.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    return fig


def plot_barycenter_progression(
    matrices: np.ndarray,
    weights_list: Optional[List[np.ndarray]] = None,
    metric: str = 'frobenius',
    n_steps: int = 6,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot progression of Wasserstein barycenters with different weights.

    Useful for understanding how different network states combine.

    Parameters
    ----------
    matrices : np.ndarray, shape (n_states, n_regions, n_regions)
        Input connectivity states
    weights_list : list of np.ndarray, optional
        List of weight vectors for computing barycenters
        (default: linear interpolation from state 0 to state 1)
    metric : str
        Wasserstein metric
    n_steps : int
        Number of barycenter steps to show
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        The figure object

    Notes
    -----
    Barycenters represent weighted averages in Wasserstein space.
    Shows how network configuration smoothly transitions between states.

    Examples
    --------
    >>> # Show transition from state 0 to state 1
    >>> states = generate_correlation_states(2, 50)
    >>> fig = plot_barycenter_progression(states, n_steps=8)
    """
    from ot_brain_dynamics.optimal_transport import wasserstein_barycenter_spd

    n_states = len(matrices)

    # Default: interpolate between first two states
    if weights_list is None:
        if n_states < 2:
            raise ValueError("Need at least 2 states for barycenter progression")

        alphas = np.linspace(0, 1, n_steps)
        weights_list = []
        for alpha in alphas:
            w = np.zeros(n_states)
            w[0] = 1 - alpha
            w[1] = alpha
            weights_list.append(w)

    # Compute barycenters
    barycenters = []
    for weights in weights_list:
        barycenter, _ = wasserstein_barycenter_spd(matrices, weights, metric=metric)
        barycenters.append(barycenter)

    # Plot
    fig, axes = plt.subplots(1, len(barycenters), figsize=figsize)

    for i, (barycenter, weights) in enumerate(zip(barycenters, weights_list)):
        ax = axes[i]
        im = ax.imshow(barycenter, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Create title showing weights
        weight_str = ', '.join([f'{w:.2f}' for w in weights[:3]])  # Show first 3
        if len(weights) > 3:
            weight_str += '...'
        ax.set_title(f'w=[{weight_str}]', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Colorbar
    fig.colorbar(im, ax=axes, label='Correlation', shrink=0.8)
    fig.suptitle('Wasserstein Barycenter Progression', fontsize=14, y=1.02)

    plt.tight_layout()

    return fig
