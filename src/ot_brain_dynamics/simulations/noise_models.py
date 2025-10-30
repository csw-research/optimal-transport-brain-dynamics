"""
Realistic noise models for fMRI simulations.

Implements physiological noise, motion artifacts, and scanner noise
to create realistic synthetic fMRI data.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import signal


def add_physiological_noise(
    time_series: np.ndarray,
    respiratory_freq: float = 0.25,
    cardiac_freq: float = 1.0,
    noise_amplitude: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add physiological noise (respiratory and cardiac) to time series.

    Simulates confounds from breathing and heartbeat that contaminate
    fMRI signals.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Clean time series data
    respiratory_freq : float, default=0.25
        Respiratory frequency in Hz (~15 breaths/min)
    cardiac_freq : float, default=1.0
        Cardiac frequency in Hz (~60 bpm)
    noise_amplitude : float, default=0.1
        Amplitude of physiological noise relative to signal
    random_state : int, optional
        Random seed

    Returns
    -------
    noisy_series : np.ndarray, shape (n_timepoints, n_regions)
        Time series with physiological noise added

    Notes
    -----
    Physiological noise has spatially correlated structure - different
    regions show phase-shifted versions of the same oscillations.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_timepoints, n_regions = time_series.shape
    t = np.arange(n_timepoints)

    # Generate respiratory oscillation
    resp_signal = np.sin(2 * np.pi * respiratory_freq * t)

    # Generate cardiac oscillation
    cardiac_signal = np.sin(2 * np.pi * cardiac_freq * t)

    # Combine physiological signals with spatial variation
    physio_noise = np.zeros_like(time_series)
    for i in range(n_regions):
        # Random phase shift and amplitude for each region
        resp_phase = np.random.rand() * 2 * np.pi
        cardiac_phase = np.random.rand() * 2 * np.pi
        resp_amp = np.random.rand() * 0.5 + 0.5
        cardiac_amp = np.random.rand() * 0.5 + 0.5

        physio_noise[:, i] = (
            resp_amp * np.sin(2 * np.pi * respiratory_freq * t + resp_phase) +
            cardiac_amp * np.sin(2 * np.pi * cardiac_freq * t + cardiac_phase)
        )

    # Scale by noise amplitude
    signal_std = time_series.std(axis=0, keepdims=True)
    physio_noise = physio_noise * noise_amplitude * signal_std

    return time_series + physio_noise


def add_motion_artifacts(
    time_series: np.ndarray,
    n_motion_events: int = 5,
    motion_amplitude: float = 0.5,
    spatial_extent: float = 0.3,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add realistic motion artifacts to time series.

    Simulates sudden head movements that cause transient signal changes
    in spatially localized regions.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Clean time series data
    n_motion_events : int, default=5
        Number of motion events
    motion_amplitude : float, default=0.5
        Amplitude of motion artifacts
    spatial_extent : float, default=0.3
        Proportion of regions affected by each motion event
    random_state : int, optional
        Random seed

    Returns
    -------
    noisy_series : np.ndarray, shape (n_timepoints, n_regions)
        Time series with motion artifacts

    Notes
    -----
    Motion artifacts:
    - Occur at random times
    - Affect random subsets of regions
    - Have rapid onset and exponential decay
    - Can be positive or negative deflections
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_timepoints, n_regions = time_series.shape
    noisy_series = time_series.copy()

    # Generate motion events
    motion_times = np.random.choice(
        range(10, n_timepoints - 10),
        size=n_motion_events,
        replace=False
    )

    for motion_t in motion_times:
        # Select affected regions
        n_affected = int(spatial_extent * n_regions)
        affected_regions = np.random.choice(
            n_regions, size=n_affected, replace=False
        )

        # Generate motion artifact with exponential decay
        decay_rate = 0.3
        duration = 20
        artifact_time = np.arange(duration)
        artifact_profile = np.exp(-decay_rate * artifact_time)

        for region in affected_regions:
            # Random amplitude and direction
            amp = np.random.randn() * motion_amplitude
            signal_std = time_series[:, region].std()

            # Apply artifact
            end_t = min(motion_t + duration, n_timepoints)
            actual_duration = end_t - motion_t
            noisy_series[motion_t:end_t, region] += (
                amp * signal_std * artifact_profile[:actual_duration]
            )

    return noisy_series


def add_scanner_noise(
    time_series: np.ndarray,
    snr: float = 2.0,
    drift_amplitude: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add scanner noise (thermal noise + drift) to time series.

    Simulates white noise from scanner electronics and slow drift
    from scanner heating/instability.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Clean time series data
    snr : float, default=2.0
        Signal-to-noise ratio (higher = less noise)
    drift_amplitude : float, default=0.1
        Amplitude of slow drift relative to signal
    random_state : int, optional
        Random seed

    Returns
    -------
    noisy_series : np.ndarray, shape (n_timepoints, n_regions)
        Time series with scanner noise

    Notes
    -----
    Scanner noise includes:
    - White Gaussian thermal noise
    - Slow polynomial drift (quadratic)
    - Independent across regions but consistent temporal structure
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_timepoints, n_regions = time_series.shape

    # Add white noise
    signal_std = time_series.std(axis=0, keepdims=True)
    noise_std = signal_std / snr
    white_noise = np.random.randn(n_timepoints, n_regions) * noise_std

    # Add slow drift (polynomial)
    t = np.linspace(-1, 1, n_timepoints)
    drift = np.zeros_like(time_series)

    for i in range(n_regions):
        # Random quadratic drift coefficients
        a = np.random.randn() * drift_amplitude
        b = np.random.randn() * drift_amplitude * 0.5
        drift[:, i] = (a * t ** 2 + b * t) * signal_std[0, i]

    return time_series + white_noise + drift


def add_realistic_fmri_noise(
    time_series: np.ndarray,
    snr: float = 2.0,
    physio_amplitude: float = 0.1,
    n_motion_events: int = 3,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add comprehensive realistic fMRI noise combining all sources.

    Convenience function that applies all noise types with typical parameters.

    Parameters
    ----------
    time_series : np.ndarray, shape (n_timepoints, n_regions)
        Clean time series data
    snr : float, default=2.0
        Signal-to-noise ratio
    physio_amplitude : float, default=0.1
        Physiological noise amplitude
    n_motion_events : int, default=3
        Number of motion events
    random_state : int, optional
        Random seed

    Returns
    -------
    noisy_series : np.ndarray, shape (n_timepoints, n_regions)
        Realistic noisy fMRI time series

    Examples
    --------
    >>> from ot_brain_dynamics.simulations import generate_var_transitions
    >>> data, states = generate_var_transitions(50, 1000)
    >>> noisy_data = add_realistic_fmri_noise(data, snr=2.0)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Apply noise sources sequentially
    noisy_series = add_scanner_noise(
        time_series, snr=snr, random_state=random_state
    )
    noisy_series = add_physiological_noise(
        noisy_series,
        noise_amplitude=physio_amplitude,
        random_state=random_state + 1 if random_state else None
    )
    noisy_series = add_motion_artifacts(
        noisy_series,
        n_motion_events=n_motion_events,
        random_state=random_state + 2 if random_state else None
    )

    return noisy_series


def generate_hemodynamic_response(
    neural_activity: np.ndarray,
    tr: float = 2.0,
    delay: float = 6.0,
    rise_time: float = 3.0,
    fall_time: float = 10.0
) -> np.ndarray:
    """
    Convolve neural activity with hemodynamic response function (HRF).

    Models the sluggish BOLD response to underlying neural activity.

    Parameters
    ----------
    neural_activity : np.ndarray, shape (n_timepoints, n_regions)
        Fast neural activity time series
    tr : float, default=2.0
        Repetition time in seconds (fMRI sampling rate)
    delay : float, default=6.0
        Peak delay in seconds
    rise_time : float, default=3.0
        Time to rise in seconds
    fall_time : float, default=10.0
        Time to fall in seconds

    Returns
    -------
    bold_signal : np.ndarray, shape (n_timepoints, n_regions)
        BOLD fMRI signal after HRF convolution

    Notes
    -----
    Uses double-gamma HRF model:
        HRF(t) = (t/d)^a * exp(-(t-d)/b) - c * (t/d2)^a2 * exp(-(t-d2)/b2)

    This models the characteristic shape of BOLD response with peak at ~6s
    and post-stimulus undershoot.
    """
    n_timepoints, n_regions = neural_activity.shape

    # Create HRF kernel (double gamma)
    t_hrf = np.arange(0, 30, tr)  # 30 seconds of HRF

    # Positive gamma
    a1, b1 = 6.0, 1.0
    gamma1 = (t_hrf / delay) ** a1 * np.exp(-(t_hrf - delay) / rise_time)

    # Negative gamma (undershoot)
    a2, b2 = 12.0, 1.0
    gamma2 = (t_hrf / (delay + 2)) ** a2 * np.exp(-(t_hrf - delay - 2) / fall_time)

    hrf = gamma1 - 0.35 * gamma2
    hrf = hrf / hrf.sum()  # Normalize

    # Convolve each region
    bold_signal = np.zeros_like(neural_activity)
    for i in range(n_regions):
        bold_signal[:, i] = np.convolve(neural_activity[:, i], hrf, mode='same')

    return bold_signal
