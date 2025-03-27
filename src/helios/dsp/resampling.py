"""Resampling functions for Helios DSP."""

import numpy as np
from typing import Optional

def upsample(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Upsample signal by integer factor.
    
    Args:
        x: Input signal
        factor: Upsampling factor
        
    Returns:
        Upsampled signal
    """
    return np.repeat(x, factor)

def downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample signal by integer factor.
    
    Args:
        x: Input signal
        factor: Downsampling factor
        
    Returns:
        Downsampled signal
    """
    return x[::factor]

def resample(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample signal to new sampling rate.
    
    Args:
        x: Input signal
        orig_fs: Original sampling frequency
        target_fs: Target sampling frequency
        
    Returns:
        Resampled signal
    """
    from scipy import signal
    factor = target_fs / orig_fs
    resampled = signal.resample(x, int(len(x) * factor))
    return np.asarray(resampled)
