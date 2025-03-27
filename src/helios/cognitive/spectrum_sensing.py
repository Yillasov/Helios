"""Spectrum sensing capabilities for cognitive radio."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

from helios.cognitive.data_structures import EnvironmentState, SpectrumBand, SpectrumOccupancy
from helios.utils.logger import get_logger
from helios.dsp.spectral import apply_fft

logger = get_logger(__name__)

class SpectrumSensor:
    """Performs spectrum sensing to detect signal presence and interference."""
    
    def __init__(self, sampling_rate: float = 1e6, fft_size: int = 1024):
        """
        Initialize the spectrum sensor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            fft_size: FFT size for spectral analysis
        """
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.noise_floor = -100.0  # dBm, default value
    
    def analyze_spectrum(self, samples: np.ndarray, center_freq: float) -> EnvironmentState:
        """
        Analyze spectrum from complex samples.
        
        Args:
            samples: Complex time-domain samples
            center_freq: Center frequency of the samples in Hz
            
        Returns:
            EnvironmentState with spectrum analysis results
        """
        # Compute power spectral density
        spectrum = self._compute_psd(samples)
        
        # Calculate frequency axis
        freq_axis = np.fft.fftshift(np.fft.fftfreq(len(spectrum), 1/self.sampling_rate))
        freq_axis += center_freq  # Shift to actual center frequency
        
        # Create environment state
        env_state = EnvironmentState(timestamp=0.0)  # Timestamp would be set by caller
        
        # Estimate noise floor
        self.noise_floor = self._estimate_noise_floor(spectrum)
        env_state.overall_noise_level = self.noise_floor
        
        # Detect signals and create spectrum bands
        bands = self._detect_signals(spectrum, freq_axis)
        
        # Add bands to environment state
        for i, (band, power) in enumerate(bands):
            band_id = f"band_{i}"
            occupancy = SpectrumOccupancy(
                band=band,
                power_level=power,
                noise_floor=self.noise_floor
            )
            env_state.spectrum_occupancy[band_id] = occupancy
        
        # Calculate overall metrics
        if env_state.spectrum_occupancy:
            env_state.average_snr = float(np.mean([occ.snr for occ in env_state.spectrum_occupancy.values()]))
            env_state.detected_signals = sum(1 for occ in env_state.spectrum_occupancy.values() if occ.is_occupied)
        
        return env_state
    
    def _compute_psd(self, samples: np.ndarray) -> np.ndarray:
        """Compute power spectral density from time samples."""
        # Apply window to reduce spectral leakage
        windowed = samples * np.hanning(len(samples))
        
        # Compute FFT
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(windowed, self.fft_size))**2)
        
        # Convert to dBm (assuming samples are voltage-like)
        spectrum_db = 10 * np.log10(spectrum + 1e-10)
        
        return spectrum_db
    
    def _estimate_noise_floor(self, spectrum: np.ndarray) -> float:
        """Estimate noise floor from spectrum."""
        # Simple approach: use lower percentile of spectrum values
        return float(np.percentile(spectrum, 10))
    
    def _detect_signals(self, spectrum: np.ndarray, freq_axis: np.ndarray) -> List[Tuple[SpectrumBand, float]]:
        """
        Detect signals in the spectrum and return as bands.
        
        Returns:
            List of (SpectrumBand, power_level) tuples
        """
        # Simple approach: divide spectrum into equal bands
        num_bands = 10
        band_size = len(spectrum) // num_bands
        
        bands = []
        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < num_bands - 1 else len(spectrum)
            
            band_spectrum = spectrum[start_idx:end_idx]
            band_freqs = freq_axis[start_idx:end_idx]
            
            # Create band
            band = SpectrumBand(
                start_freq=band_freqs[0],
                end_freq=band_freqs[-1]
            )
            
            # Calculate average power in band
            avg_power = np.mean(band_spectrum)
            
            bands.append((band, avg_power))
        
        return bands