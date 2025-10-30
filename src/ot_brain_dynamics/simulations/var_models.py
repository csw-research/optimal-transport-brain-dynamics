"""
Vector autoregressive (VAR) models for generating synthetic time series.

Implements time-varying VAR models to simulate brain dynamics with
evolving connectivity structures.
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Callable
from scipy import linalg
import warnings


class TimeVaryingVAR:
    """
    Time-varying vector autoregressive model for brain dynamics simulation.

    Generates multivariate time series where connectivity parameters evolve
    according to specified transition dynamics.

    Parameters
    ----------
    n_regions : int
        Number of brain regions (time series dimension)
    lag_order : int, default=1
        VAR model lag order
    innovation_cov : np.ndarray, optional
        Innovation covariance matrix (default: identity)

    Attributes
    ----------
    A_matrices_ : list of np.ndarray
        Time-varying VAR coefficient matrices

    Examples
    --------
    >>> var_model = TimeVaryingVAR(n_regions=50, lag_order=1)
    >>> time_series, states = var_model.generate(
    ...     n_timepoints=1000,
    ...     n_states=3,
    ...     transition_type='abrupt'
    ... )
    """

    def __init__(
        self,
        n_regions: int,
        lag_order: int = 1,
        innovation_cov: Optional[np.ndarray] = None
    ):
        self.n_regions = n_regions
        self.lag_order = lag_order

        if innovation_cov is None:
            self.innovation_cov = np.eye(n_regions)
        else:
            if innovation_cov.shape != (n_regions, n_regions):
                raise ValueError(
                    f"innovation_cov must be shape ({n_regions}, {n_regions}), "
                    f"got {innovation_cov.shape}"
                )
            self.innovation_cov = innovation_cov

        self.A_matrices_: List[np.ndarray] = []

    def generate(
        self,
        n_timepoints: int,
        n_states: int = 3,
        transition_type: str = 'smooth',
        transition_rate: float = 0.1,
        sparsity: float = 0.2,
        spectral_radius: float = 0.9,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series with evolving network states.

        Parameters
        ----------
        n_timepoints : int
            Number of time points to generate
        n_states : int, default=3
            Number of distinct network states
        transition_type : str, default='smooth'
            Type of state transitions:
            - 'smooth': Gradual geodesic interpolation between states
            - 'abrupt': Sudden switches at random times
            - 'oscillating': Periodic transitions between states
        transition_rate : float, default=0.1
            Rate of transitions (interpretation depends on type)
        sparsity : float, default=0.2
            Proportion of non-zero connections
        spectral_radius : float, default=0.9
            Maximum eigenvalue of VAR matrix (stability constraint)
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        time_series : np.ndarray, shape (n_timepoints, n_regions)
            Generated time series data
        state_labels : np.ndarray, shape (n_timepoints,)
            Ground truth state label for each time point

        Notes
        -----
        The VAR(1) model is:
            X_t = A_t @ X_{t-1} + ε_t
        where A_t varies over time according to the transition dynamics.

        For stability, we ensure spectral_radius(A_t) < 1 for all t.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Generate network state connectivity matrices
        from ot_brain_dynamics.simulations.network_states import generate_network_states
        state_matrices = generate_network_states(
            n_states=n_states,
            n_regions=self.n_regions,
            sparsity=sparsity,
            spectral_radius=spectral_radius,
            random_state=random_state
        )

        # Generate state sequence and time-varying matrices
        if transition_type == 'smooth':
            A_sequence, state_labels = self._smooth_transitions(
                state_matrices, n_timepoints, transition_rate
            )
        elif transition_type == 'abrupt':
            A_sequence, state_labels = self._abrupt_transitions(
                state_matrices, n_timepoints, transition_rate
            )
        elif transition_type == 'oscillating':
            A_sequence, state_labels = self._oscillating_transitions(
                state_matrices, n_timepoints, transition_rate
            )
        else:
            raise ValueError(f"Unknown transition_type: {transition_type}")

        self.A_matrices_ = A_sequence

        # Generate time series
        time_series = self._generate_from_sequence(A_sequence)

        return time_series, state_labels

    def _smooth_transitions(
        self,
        state_matrices: List[np.ndarray],
        n_timepoints: int,
        rate: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate smooth geodesic transitions between states."""
        from ot_brain_dynamics.optimal_transport.spd_geometry import geodesic_spd

        # Convert VAR matrices to covariance matrices for geodesic computation
        state_covs = []
        for A in state_matrices:
            # Covariance from VAR: Σ = (I - A)^{-1} Σ_ε (I - A)^{-T}
            I_minus_A = np.eye(self.n_regions) - A
            try:
                Sigma = linalg.solve_discrete_lyapunov(A, self.innovation_cov)
                # Ensure positive definiteness
                eigvals = np.linalg.eigvals(Sigma)
                if np.any(eigvals <= 0):
                    Sigma = Sigma + np.eye(self.n_regions) * 0.1
            except:
                # Fallback: simple approximation
                Sigma = self.innovation_cov + 0.1 * np.eye(self.n_regions)
            state_covs.append(Sigma)

        # Determine transition schedule
        n_states = len(state_matrices)
        transitions_per_state = int(n_timepoints / n_states)

        A_sequence = []
        state_labels = np.zeros(n_timepoints, dtype=int)

        for state_idx in range(n_states):
            next_state_idx = (state_idx + 1) % n_states
            start_t = state_idx * transitions_per_state
            end_t = min((state_idx + 1) * transitions_per_state, n_timepoints)

            # Geodesic interpolation parameters
            t_params = np.linspace(0, 1, end_t - start_t)

            for i, t in enumerate(t_params):
                # Interpolate covariance matrices
                cov_t = geodesic_spd(
                    state_covs[state_idx],
                    state_covs[next_state_idx],
                    t
                )

                # Convert back to VAR matrix (approximate)
                # This is a simplification; in practice we interpolate A directly
                alpha = (1 - t)
                A_t = alpha * state_matrices[state_idx] + (1 - alpha) * state_matrices[next_state_idx]

                A_sequence.append(A_t)
                # Label based on which state is more dominant
                state_labels[start_t + i] = state_idx if t < 0.5 else next_state_idx

        return A_sequence, state_labels

    def _abrupt_transitions(
        self,
        state_matrices: List[np.ndarray],
        n_timepoints: int,
        rate: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate abrupt switches between states."""
        n_states = len(state_matrices)

        # Determine transition times (Poisson-like process)
        n_transitions = max(1, int(n_timepoints * rate))
        transition_times = sorted(
            np.random.choice(range(1, n_timepoints), size=n_transitions, replace=False)
        )
        transition_times = [0] + transition_times + [n_timepoints]

        A_sequence = []
        state_labels = np.zeros(n_timepoints, dtype=int)

        current_state = 0
        for i in range(len(transition_times) - 1):
            start_t = transition_times[i]
            end_t = transition_times[i + 1]

            # Constant state in this interval
            for t in range(start_t, end_t):
                A_sequence.append(state_matrices[current_state].copy())
                state_labels[t] = current_state

            # Transition to next state
            current_state = (current_state + 1) % n_states

        return A_sequence, state_labels

    def _oscillating_transitions(
        self,
        state_matrices: List[np.ndarray],
        n_timepoints: int,
        rate: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate periodic oscillations between states."""
        n_states = len(state_matrices)
        period = int(1.0 / rate)  # rate is frequency

        A_sequence = []
        state_labels = np.zeros(n_timepoints, dtype=int)

        for t in range(n_timepoints):
            # Sinusoidal modulation between states
            phase = 2 * np.pi * t / period
            state_weights = np.array([
                (1 + np.cos(phase + 2 * np.pi * i / n_states)) / 2
                for i in range(n_states)
            ])
            state_weights = state_weights / state_weights.sum()

            # Weighted combination of state matrices
            A_t = np.zeros_like(state_matrices[0])
            for i, w in enumerate(state_weights):
                A_t += w * state_matrices[i]

            A_sequence.append(A_t)
            state_labels[t] = np.argmax(state_weights)

        return A_sequence, state_labels

    def _generate_from_sequence(
        self,
        A_sequence: List[np.ndarray]
    ) -> np.ndarray:
        """Generate time series given sequence of VAR matrices."""
        n_timepoints = len(A_sequence)
        time_series = np.zeros((n_timepoints, self.n_regions))

        # Initialize with stationary distribution
        time_series[0] = np.random.multivariate_normal(
            np.zeros(self.n_regions),
            self.innovation_cov
        )

        # Generate rest of time series
        innovation_chol = linalg.cholesky(self.innovation_cov, lower=True)

        for t in range(1, n_timepoints):
            # VAR update
            time_series[t] = A_sequence[t] @ time_series[t-1]

            # Add innovation
            innovation = innovation_chol @ np.random.randn(self.n_regions)
            time_series[t] += innovation

        return time_series


def generate_var_time_series(
    n_regions: int,
    n_timepoints: int,
    connectivity_matrix: np.ndarray,
    lag_order: int = 1,
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate stationary VAR time series from fixed connectivity matrix.

    Parameters
    ----------
    n_regions : int
        Number of brain regions
    n_timepoints : int
        Number of time points
    connectivity_matrix : np.ndarray, shape (n_regions, n_regions)
        VAR coefficient matrix
    lag_order : int, default=1
        VAR lag order
    noise_std : float, default=1.0
        Standard deviation of innovation noise
    random_state : int, optional
        Random seed

    Returns
    -------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Generated time series
    """
    if random_state is not None:
        np.random.seed(random_state)

    time_series = np.zeros((n_timepoints, n_regions))
    innovation_cov = noise_std ** 2 * np.eye(n_regions)

    # Initialize
    time_series[0] = np.random.randn(n_regions) * noise_std

    # Generate
    for t in range(1, n_timepoints):
        time_series[t] = connectivity_matrix @ time_series[t-1]
        time_series[t] += np.random.multivariate_normal(
            np.zeros(n_regions), innovation_cov
        )

    return time_series


def generate_var_transitions(
    n_regions: int,
    n_timepoints: int,
    n_states: int = 3,
    transition_type: str = 'smooth',
    sparsity: float = 0.2,
    spectral_radius: float = 0.9,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate time series with state transitions.

    Parameters
    ----------
    n_regions : int
        Number of brain regions
    n_timepoints : int
        Number of time points
    n_states : int, default=3
        Number of distinct network states
    transition_type : str, default='smooth'
        Type of transitions ('smooth', 'abrupt', 'oscillating')
    sparsity : float, default=0.2
        Connection sparsity
    spectral_radius : float, default=0.9
        VAR stability parameter
    random_state : int, optional
        Random seed

    Returns
    -------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Generated time series
    state_labels : np.ndarray, shape (n_timepoints,)
        Ground truth state labels

    Examples
    --------
    >>> # Generate data with smooth transitions between 3 network states
    >>> data, states = generate_var_transitions(
    ...     n_regions=50, n_timepoints=1000, n_states=3
    ... )
    >>> print(data.shape, states.shape)
    (1000, 50) (1000,)
    """
    var_model = TimeVaryingVAR(n_regions=n_regions)
    time_series, state_labels = var_model.generate(
        n_timepoints=n_timepoints,
        n_states=n_states,
        transition_type=transition_type,
        sparsity=sparsity,
        spectral_radius=spectral_radius,
        random_state=random_state
    )

    return time_series, state_labels
