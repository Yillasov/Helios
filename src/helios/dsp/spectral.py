"""Spectral analysis functions for Helios DSP."""

import numpy as np
from typing import Tuple, Optional

def apply_fft(x: np.ndarray, fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Fast Fourier Transform to signal.
    
    Args:
        x: Input signal
        fs: Sampling frequency (if provided, returns frequency axis)
        
    Returns:
        Tuple of (frequencies, spectrum)
    """
    spectrum = np.fft.fftshift(np.fft.fft(x))
    
    if fs is not None:
        freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1/fs))
        return freqs, spectrum
    else:
        return np.arange(len(spectrum)), spectrum

def apply_ifft(X: np.ndarray) -> np.ndarray:
    """
    Apply Inverse Fast Fourier Transform.
    
    Args:
        X: Input spectrum
        
    Returns:
        Time domain signal
    """
    return np.fft.ifft(np.fft.ifftshift(X))

def power_spectrum(x: np.ndarray, fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectrum of signal.
    
    Args:
        x: Input signal
        fs: Sampling frequency
        
    Returns:
        Tuple of (frequencies, power)
    """
    freqs, spectrum = apply_fft(x, fs)
    power = np.abs(spectrum)**2 / len(x)
    return freqs, power