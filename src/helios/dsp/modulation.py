"""Modulation and demodulation functions for Helios DSP."""

import numpy as np
from typing import Dict, Any, Tuple

from helios.core.data_structures import ModulationType, Waveform

def modulate_am(carrier: np.ndarray, message: np.ndarray, mod_index: float = 0.5) -> np.ndarray:
    """
    Apply amplitude modulation.
    
    Args:
        carrier: Carrier signal
        message: Message signal
        mod_index: Modulation index
        
    Returns:
        Modulated signal
    """
    # Ensure message is properly scaled
    message = message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message
    return carrier * (1 + mod_index * message)

def demodulate_am(signal: np.ndarray) -> np.ndarray:
    """
    Demodulate AM signal using envelope detection.
    
    Args:
        signal: Modulated signal
        
    Returns:
        Demodulated message
    """
    # Simple envelope detector
    analytic_signal = np.abs(signal)
    # Remove DC component
    return analytic_signal - np.mean(analytic_signal)

def modulate_fm(carrier: np.ndarray, message: np.ndarray, 
                mod_index: float = 0.5, fs: float = 1.0) -> np.ndarray:
    """
    Apply frequency modulation.
    
    Args:
        carrier: Carrier signal
        message: Message signal
        mod_index: Modulation index
        fs: Sampling frequency
        
    Returns:
        Modulated signal
    """
    # Normalize message
    message = message / np.max(np.abs(message)) if np.max(np.abs(message)) > 0 else message
    
    # Integrate message to get phase
    phase = np.cumsum(message) * mod_index / fs
    
    # Apply phase modulation
    return np.exp(1j * phase) * carrier

def modulate_psk(bits: np.ndarray, n_bits: int = 2) -> np.ndarray:
    """
    Apply Phase Shift Keying modulation.
    
    Args:
        bits: Input bit stream
        n_bits: Number of bits per symbol (1=BPSK, 2=QPSK, etc.)
        
    Returns:
        Complex baseband modulated signal
    """
    # Group bits into symbols
    symbols = np.reshape(bits[:len(bits) - len(bits) % n_bits], (-1, n_bits))
    
    # Convert to symbol values
    symbol_values = np.packbits(symbols, axis=1).flatten()
    
    # Map to constellation points
    n_symbols = 2**n_bits
    phases = symbol_values * 2 * np.pi / n_symbols
    
    # Generate complex signal
    return np.exp(1j * phases)