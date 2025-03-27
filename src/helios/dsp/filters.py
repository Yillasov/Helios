"""Digital filter implementations for Helios DSP."""

import numpy as np
from typing import List, Union, Optional

def fir_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Apply FIR filter to input signal.
    
    Args:
        x: Input signal
        h: Filter coefficients
        
    Returns:
        Filtered signal
    """
    return np.convolve(x, h, mode='same')

def iir_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Apply IIR filter to input signal.
    
    Args:
        x: Input signal
        b: Numerator coefficients
        a: Denominator coefficients
        
    Returns:
        Filtered signal
    """
    from scipy import signal
    result = signal.lfilter(b, a, x)
    # Handle the case where lfilter returns a tuple
    if isinstance(result, tuple):
        return result[0]  # Return only the filtered signal, not the filter state
    return result

def lowpass_filter(x: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply lowpass filter to input signal.
    
    Args:
        x: Input signal
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    from scipy import signal
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Handle potential issues with butter function
    butter_result = signal.butter(order, normal_cutoff, btype='low')
    if butter_result is None:
        raise ValueError("Failed to create filter coefficients")
    
    # Ensure we have exactly two elements (b, a)
    if isinstance(butter_result, tuple) and len(butter_result) >= 2:
        b, a = butter_result[0], butter_result[1]
    else:
        raise ValueError(f"Unexpected return format from butter: {type(butter_result)}")
    
    result = signal.lfilter(b, a, x)
    # Handle the case where lfilter returns a tuple
    if isinstance(result, tuple):
        return result[0]
    return result