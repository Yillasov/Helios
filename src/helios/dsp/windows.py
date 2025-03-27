"""Window functions for Helios DSP."""

import numpy as np
from typing import Literal, Optional

WindowType = Literal['rectangular', 'hamming', 'hanning', 'blackman', 'kaiser']

def get_window(window_type: WindowType, length: int, beta: Optional[float] = None) -> np.ndarray:
    """
    Generate window function.
    
    Args:
        window_type: Type of window
        length: Window length
        beta: Kaiser window parameter (only used for kaiser window)
        
    Returns:
        Window function
    """
    if window_type == 'rectangular':
        return np.ones(length)
    elif window_type == 'hamming':
        return np.hamming(length)
    elif window_type == 'hanning':
        return np.hanning(length)
    elif window_type == 'blackman':
        return np.blackman(length)
    elif window_type == 'kaiser':
        return np.kaiser(length, beta or 14)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

def apply_window(x: np.ndarray, window_type: WindowType) -> np.ndarray:
    """
    Apply window function to signal.
    
    Args:
        x: Input signal
        window_type: Type of window
        
    Returns:
        Windowed signal
    """
    window = get_window(window_type, len(x))
    return x * window