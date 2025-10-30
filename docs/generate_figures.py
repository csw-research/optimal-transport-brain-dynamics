"""
Generate example figures for documentation.

This script creates representative visualizations showcasing the package's
capabilities for inclusion in README and documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from ot_brain_dynamics.simulations import generate_var_transitions, generate_correlation_states
from ot_brain_dynamics.connectivity import (
    sliding_window_connectivity,
    wasserstein_trajectory,
    detect_state_transitions,
    segment_network_states,
)
from ot_brain_dynamics.visualization import (
    plot_connectivity_evolution,
    plot_wasserstein_trajectory,
    plot_state_timeline,
    plot_distance_matrix,
    plot_geodesic_path,
    plot_manifold_embedding,
)

# Set random seed for reproducibility
np.random.seed(42)

print("Generating example figures for documentation...")

# ============================================================================
# Figure 1: Network state transitions over time
# ============================================================================
print("\n[1/6] Generating network evolution figure...")

n_regions = 50
n_timepoints = 800

time_series, true_states = generate_var_transitions(
    n_regions=n_regions,
    n_timepoints=n_timepoints,
    n_states=3,
    transition_type='smooth',
    random_state=42
)

conn_matrices = sliding_window_connectivity(time_series, window_size=50)

fig = plot_connectivity_evolution(conn_matrices, n_plots=6, figsize=(16, 8))
plt.savefig('docs/figures/connectivity_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: connectivity_evolution.png")

# ============================================================================
# Figure 2: Wasserstein distance trajectory
# ============================================================================
print("\n[2/6] Generating Wasserstein trajectory figure...")

distances = wasserstein_trajectory(conn_matrices)
transitions = detect_state_transitions(distances, method='peaks')

fig = plot_wasserstein_trajectory(distances, transitions=transitions, smoothing=5)
plt.savefig('docs/figures/wasserstein_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: wasserstein_trajectory.png")

# ============================================================================
# Figure 3: State segmentation timeline
# ============================================================================
print("\n[3/6] Generating state timeline figure...")

state_labels = segment_network_states(conn_matrices, n_states=3, method='kmeans')

fig = plot_state_timeline(state_labels, distances=distances)
plt.savefig('docs/figures/state_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: state_timeline.png")

# ============================================================================
# Figure 4: Pairwise distance matrix
# ============================================================================
print("\n[4/6] Generating distance matrix figure...")

_, distance_matrix = wasserstein_trajectory(conn_matrices, return_matrices=True)

fig = plot_distance_matrix(distance_matrix, state_labels=state_labels)
plt.savefig('docs/figures/distance_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: distance_matrix.png")

# ============================================================================
# Figure 5: Geodesic path between states
# ============================================================================
print("\n[5/6] Generating geodesic path figure...")

# Generate two distinct network states
states = generate_correlation_states(n_states=2, n_regions=50, random_state=42)

fig = plot_geodesic_path(states[0], states[1], n_steps=8, figsize=(16, 3))
plt.savefig('docs/figures/geodesic_path.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: geodesic_path.png")

# ============================================================================
# Figure 6: Manifold embedding
# ============================================================================
print("\n[6/6] Generating manifold embedding figure...")

fig = plot_manifold_embedding(
    conn_matrices[::2],  # Subsample for speed
    state_labels=state_labels[::2],
    method='mds'
)
plt.savefig('docs/figures/manifold_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: manifold_embedding.png")

# ============================================================================
# Figure 7: Create a combined overview figure
# ============================================================================
print("\n[Bonus] Creating overview figure combining key visualizations...")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel A: Sample connectivity matrices
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(conn_matrices[10], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_title('(A) State 1 Connectivity', fontweight='bold', fontsize=11)
ax1.set_xlabel('Region')
ax1.set_ylabel('Region')
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(conn_matrices[30], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_title('(B) State 2 Connectivity', fontweight='bold', fontsize=11)
ax2.set_xlabel('Region')
plt.colorbar(im2, ax=ax2, fraction=0.046)

ax3 = fig.add_subplot(gs[0, 2])
im3 = ax3.imshow(conn_matrices[50], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_title('(C) State 3 Connectivity', fontweight='bold', fontsize=11)
ax3.set_xlabel('Region')
plt.colorbar(im3, ax=ax3, fraction=0.046)

# Panel D: Wasserstein trajectory
ax4 = fig.add_subplot(gs[1, :2])
ax4.plot(distances, 'b-', linewidth=2, alpha=0.7)
ax4.fill_between(range(len(distances)), 0, distances, alpha=0.2, color='blue')
threshold = distances.mean() + 2 * distances.std()
ax4.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='Threshold')
for trans in transitions:
    ax4.axvline(trans, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax4.set_xlabel('Time Window', fontsize=11)
ax4.set_ylabel('Wasserstein Distance', fontsize=11)
ax4.set_title('(D) Network Evolution Trajectory', fontweight='bold', fontsize=11)
ax4.grid(alpha=0.3)
ax4.legend()

# Panel E: State timeline
ax5 = fig.add_subplot(gs[1, 2])
n_states_unique = len(np.unique(state_labels))
cmap = plt.cm.get_cmap('tab10')
state_colors = [cmap(i) for i in range(n_states_unique)]

for i in range(len(state_labels) - 1):
    color = state_colors[state_labels[i]]
    ax5.axvspan(i, i + 1, facecolor=color, alpha=0.7, edgecolor='none')

ax5.set_xlabel('Time Window', fontsize=11)
ax5.set_ylabel('Network State', fontsize=11)
ax5.set_yticks(range(n_states_unique))
ax5.set_yticklabels([f'State {i}' for i in range(n_states_unique)])
ax5.set_title('(E) State Segmentation', fontweight='bold', fontsize=11)
ax5.grid(axis='x', alpha=0.3)

# Panel F: Distance matrix
ax6 = fig.add_subplot(gs[2, :])
im6 = ax6.imshow(distance_matrix, cmap='viridis', aspect='auto', origin='lower')
ax6.set_xlabel('Time Window', fontsize=11)
ax6.set_ylabel('Time Window', fontsize=11)
ax6.set_title('(F) Pairwise Wasserstein Distance Matrix (Block Structure Reveals States)',
              fontweight='bold', fontsize=11)
plt.colorbar(im6, ax=ax6, label='Wasserstein Distance')

fig.suptitle('Optimal Transport for Dynamic Brain Connectivity: Example Results',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('docs/figures/overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: overview.png")

print("\n" + "="*80)
print("All figures generated successfully!")
print("Figures saved in: docs/figures/")
print("="*80)
