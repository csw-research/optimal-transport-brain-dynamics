"""
Generate realistic network connectivity states for simulations.

Provides functions to create biologically plausible network structures
including modular, hub-based, and random configurations.
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy import linalg


def generate_network_states(
    n_states: int,
    n_regions: int,
    sparsity: float = 0.2,
    spectral_radius: float = 0.9,
    random_state: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate multiple distinct network connectivity states.

    Parameters
    ----------
    n_states : int
        Number of distinct states to generate
    n_regions : int
        Number of brain regions
    sparsity : float, default=0.2
        Proportion of non-zero connections
    spectral_radius : float, default=0.9
        Maximum eigenvalue magnitude (for stability)
    random_state : int, optional
        Random seed

    Returns
    -------
    states : list of np.ndarray
        List of connectivity matrices, each shape (n_regions, n_regions)

    Notes
    -----
    Generates diverse network topologies:
    - State 0: Modular structure
    - State 1: Hub-based architecture
    - State 2+: Random sparse networks

    All matrices are scaled to have specified spectral radius for VAR stability.
    """
    if random_state is not None:
        np.random.seed(random_state)

    states = []

    # State 1: Modular network
    if n_states >= 1:
        n_modules = min(5, n_regions // 10)
        A = create_modular_network(n_regions, n_modules, sparsity)
        A = _scale_spectral_radius(A, spectral_radius)
        states.append(A)

    # State 2: Hub network
    if n_states >= 2:
        n_hubs = min(5, n_regions // 20)
        A = create_hub_network(n_regions, n_hubs, sparsity)
        A = _scale_spectral_radius(A, spectral_radius)
        states.append(A)

    # Additional states: Random networks
    for i in range(max(0, n_states - 2)):
        A = create_random_network(n_regions, sparsity)
        A = _scale_spectral_radius(A, spectral_radius)
        states.append(A)

    return states


def create_modular_network(
    n_regions: int,
    n_modules: int = 5,
    sparsity: float = 0.2
) -> np.ndarray:
    """
    Create a modular (block-diagonal-like) network structure.

    Simulates functionally segregated brain networks where regions are
    organized into modules with dense within-module connections and
    sparse between-module connections.

    Parameters
    ----------
    n_regions : int
        Total number of brain regions
    n_modules : int, default=5
        Number of modules
    sparsity : float, default=0.2
        Overall connection sparsity

    Returns
    -------
    A : np.ndarray, shape (n_regions, n_regions)
        Modular connectivity matrix

    Notes
    -----
    Within-module connections are 5x more likely than between-module connections.
    """
    A = np.zeros((n_regions, n_regions))

    # Assign regions to modules
    module_size = n_regions // n_modules
    module_assignments = np.repeat(range(n_modules), module_size)
    if len(module_assignments) < n_regions:
        module_assignments = np.concatenate([
            module_assignments,
            np.full(n_regions - len(module_assignments), n_modules - 1)
        ])

    # Generate connections
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue

            # Within-module connections are denser
            if module_assignments[i] == module_assignments[j]:
                prob = 5 * sparsity
            else:
                prob = sparsity / 5

            if np.random.rand() < prob:
                A[i, j] = np.random.randn() * 0.3

    return A


def create_hub_network(
    n_regions: int,
    n_hubs: int = 5,
    sparsity: float = 0.2
) -> np.ndarray:
    """
    Create a hub-based network with highly connected hub nodes.

    Simulates brain networks with connector hubs that integrate information
    across regions, reflecting hierarchical organization.

    Parameters
    ----------
    n_regions : int
        Total number of brain regions
    n_hubs : int, default=5
        Number of hub nodes
    sparsity : float, default=0.2
        Overall connection sparsity

    Returns
    -------
    A : np.ndarray, shape (n_regions, n_regions)
        Hub network connectivity matrix

    Notes
    -----
    Hub nodes have 10x higher connection probability than peripheral nodes.
    Implements scale-free-like degree distribution.
    """
    A = np.zeros((n_regions, n_regions))

    # Designate hub nodes (first n_hubs)
    hubs = np.arange(n_hubs)
    periphery = np.arange(n_hubs, n_regions)

    # Hub-to-hub connections (dense)
    for i in hubs:
        for j in hubs:
            if i != j and np.random.rand() < 0.8:
                A[i, j] = np.random.randn() * 0.4

    # Hub-to-periphery connections (moderate)
    for i in hubs:
        for j in periphery:
            if np.random.rand() < 10 * sparsity:
                A[i, j] = np.random.randn() * 0.3
                A[j, i] = np.random.randn() * 0.3

    # Periphery-to-periphery connections (sparse)
    for i in periphery:
        for j in periphery:
            if i < j and np.random.rand() < sparsity:
                A[i, j] = np.random.randn() * 0.2
                A[j, i] = np.random.randn() * 0.2

    return A


def create_random_network(
    n_regions: int,
    sparsity: float = 0.2
) -> np.ndarray:
    """
    Create a random sparse network (Erdős-Rényi-like).

    Parameters
    ----------
    n_regions : int
        Number of brain regions
    sparsity : float, default=0.2
        Connection probability

    Returns
    -------
    A : np.ndarray, shape (n_regions, n_regions)
        Random connectivity matrix
    """
    # Generate random connections
    A = np.random.randn(n_regions, n_regions) * 0.3

    # Apply sparsity mask
    mask = np.random.rand(n_regions, n_regions) < sparsity
    A = A * mask

    # Remove self-connections
    np.fill_diagonal(A, 0)

    return A


def _scale_spectral_radius(A: np.ndarray, target_radius: float) -> np.ndarray:
    """
    Scale matrix to have specified spectral radius.

    Ensures VAR model stability by constraining largest eigenvalue magnitude.

    Parameters
    ----------
    A : np.ndarray
        Input matrix
    target_radius : float
        Desired spectral radius (should be < 1 for stability)

    Returns
    -------
    A_scaled : np.ndarray
        Scaled matrix with spectral_radius(A_scaled) = target_radius
    """
    # Compute current spectral radius
    eigenvalues = linalg.eigvals(A)
    current_radius = np.max(np.abs(eigenvalues))

    if current_radius < 1e-10:
        # Matrix is essentially zero
        return A

    # Scale to target radius
    scale_factor = target_radius / current_radius
    A_scaled = A * scale_factor

    return A_scaled


def create_transition_matrix_spd(
    n_regions: int,
    structure: str = 'modular',
    condition_number: float = 10.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Create SPD correlation/covariance matrix with specified structure.

    Useful for generating ground truth connectivity states.

    Parameters
    ----------
    n_regions : int
        Matrix dimension
    structure : str, default='modular'
        Network structure ('modular', 'hub', 'random')
    condition_number : float, default=10.0
        Matrix condition number (controls spread of eigenvalues)
    random_state : int, optional
        Random seed

    Returns
    -------
    C : np.ndarray, shape (n_regions, n_regions)
        SPD correlation matrix

    Notes
    -----
    Generates matrices by:
    1. Creating structured precision matrix (inverse covariance)
    2. Inverting to get covariance
    3. Normalizing to correlation matrix
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate precision matrix with structure
    if structure == 'modular':
        Theta = create_modular_network(n_regions, n_modules=5, sparsity=0.3)
    elif structure == 'hub':
        Theta = create_hub_network(n_regions, n_hubs=5, sparsity=0.3)
    else:
        Theta = create_random_network(n_regions, sparsity=0.3)

    # Make symmetric
    Theta = (Theta + Theta.T) / 2

    # Add diagonal dominance to ensure positive definiteness
    Theta = Theta + (np.abs(Theta).sum(axis=1) + 1.0)[:, None] * np.eye(n_regions)

    # Invert to get covariance
    try:
        Sigma = linalg.inv(Theta)
    except:
        # Fallback if inversion fails
        Sigma = linalg.inv(Theta + 0.1 * np.eye(n_regions))

    # Control condition number
    eigvals, eigvecs = linalg.eigh(Sigma)
    eigvals = eigvals / eigvals.max()  # Normalize largest eigenvalue to 1
    eigvals = np.maximum(eigvals, 1.0 / condition_number)  # Constrain smallest
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Convert to correlation matrix
    D = np.sqrt(np.diag(Sigma))
    C = Sigma / np.outer(D, D)

    # Ensure perfect symmetry and unit diagonal
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)

    return C


def generate_correlation_states(
    n_states: int,
    n_regions: int,
    condition_number: float = 10.0,
    random_state: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate multiple correlation matrix states for connectivity analysis.

    Parameters
    ----------
    n_states : int
        Number of states
    n_regions : int
        Matrix dimension
    condition_number : float, default=10.0
        Condition number for matrices
    random_state : int, optional
        Random seed

    Returns
    -------
    states : list of np.ndarray
        List of correlation matrices, each shape (n_regions, n_regions)

    Examples
    --------
    >>> states = generate_correlation_states(n_states=3, n_regions=50)
    >>> # Use for connectivity analysis
    >>> from ot_brain_dynamics.optimal_transport import wasserstein_spd
    >>> dist = wasserstein_spd(states[0], states[1])
    """
    if random_state is not None:
        np.random.seed(random_state)

    structures = ['modular', 'hub', 'random']
    states = []

    for i in range(n_states):
        structure = structures[i % len(structures)]
        C = create_transition_matrix_spd(
            n_regions,
            structure=structure,
            condition_number=condition_number,
            random_state=random_state + i if random_state else None
        )
        states.append(C)

    return states
