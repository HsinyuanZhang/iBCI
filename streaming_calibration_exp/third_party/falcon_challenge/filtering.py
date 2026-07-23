"""Neural signal filtering utilities.
Adapted from the FALCON challenge repo (https://github.com/snel-repo/falcon-challenge).
Licensed under MIT License.
"""
import scipy.signal as signal
import numpy as np


NEURAL_TAU_MS = 240.  # exponential filter from H1 Lab


def apply_exponential_filter(
        x, tau=NEURAL_TAU_MS, bin_size=20, extent: int = 1
    ):
    """
    Apply a **causal** exponential filter to the neural signal.

    :param x: NumPy array of shape (time, channels)
    :param tau: Decay rate (time constant) of the exponential filter
    :param bin_size: Bin size in ms (default is 20ms)
    :param extent: Number of time constants to extend the filter kernel
    :return: Filtered signal

    Implementation notes:
    # extent should be 3 for reporting parity, but reference hardcodes a kernel that's equivalent to extent=1
    """
    assert len(x.shape) == 2, "Still need to implement 3D convolve..."
    t = np.arange(0, extent * tau, bin_size)
    # Exponential filter kernel
    kernel = np.exp(-t / tau)
    kernel /= np.sum(kernel)
    # Apply the filter
    filtered_signal = np.array([signal.convolve(x[..., ch], kernel, mode='full')[:len(x)] for ch in range(x.shape[-1])]).T
    return filtered_signal
