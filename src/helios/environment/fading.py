"""Statistical fading models for RF propagation."""

import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from helios.core.interfaces import IPropagationModel
from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FadingParameters:
    """Parameters for statistical fading models."""
    k_factor: float = 0.0  # Rician K-factor (0 for Rayleigh)
    doppler_spread: float = 0.0  # Hz
    coherence_time: float = 1.0  # seconds
    coherence_bandwidth: float = 1e6  # Hz
    delay_spread: float = 1e-6  # seconds


class StatisticalFadingModel(IPropagationModel):
    """
    Simplified statistical fading model implementing Rayleigh and Rician fading.
    Can be used in combination with other propagation models.
    """
    
    def __init__(self, base_model: IPropagationModel):
        """
        Initialize with a base propagation model.
        
        Args:
            base_model: Base propagation model for path loss calculation
        """
        self.base_model = base_model
        self.fading_cache = {}  # Cache fading coefficients
        self.rng = np.random.RandomState()  # For reproducible random numbers

    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss with fading effects.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB including fading
        """
        # Get base path loss from underlying model
        base_path_loss = self.base_model.calculate_path_loss(
            tx_position, rx_position, frequency, environment
        )
        
        # Use getattr to safely access fading_params and get k_factor
        fading_params = getattr(environment, 'fading_params', None)
        k_factor = fading_params.k_factor if fading_params else 0.0
        
        # Apply fading component
        fading_db = self._get_fading_component(tx_position, rx_position, frequency, k_factor)
        
        # Return combined path loss
        return base_path_loss + fading_db
    
    def apply_propagation_effects(self, 
                                 signal: Signal,
                                 rx_platform: Platform,
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply fading effects to a signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            environment: Environmental parameters
        
        Returns:
            Modified signal with fading effects applied
        """
        # First apply base model effects
        modified_signal = self.base_model.apply_propagation_effects(
            signal, rx_platform, environment
        )
        
        # Use getattr to safely access fading_params and get k_factor
        fading_params = getattr(environment, 'fading_params', None)
        k_factor = fading_params.k_factor if fading_params else 0.0
        
        # Apply fading to power
        fading_db = self._get_fading_component(
            signal.origin, 
            rx_platform.position, 
            signal.waveform.center_frequency,
            k_factor
        )
        
        modified_signal.power -= fading_db
        return modified_signal
    
    def _get_fading_component(self, 
                             tx_pos: Position, 
                             rx_pos: Position, 
                             frequency: float,
                             k_factor: float) -> float:
        """
        Get fading component for given positions and parameters.
        Uses caching to avoid recalculating for the same inputs.
        
        Args:
            tx_pos: Transmitter position
            rx_pos: Receiver position
            frequency: Signal frequency in Hz
            k_factor: Rician K-factor (0 for Rayleigh)
            
        Returns:
            Fading loss in dB
        """
        # Create a cache key with reduced precision to improve cache hits
        cache_key = f"{tx_pos.x:.1f}_{tx_pos.y:.1f}_{rx_pos.x:.1f}_{rx_pos.y:.1f}_{frequency/1e6:.1f}"
        
        # Return cached value if available
        if cache_key in self.fading_cache:
            return self.fading_cache[cache_key]
        
        # Calculate fading based on model type
        if k_factor <= 0.001:
            # Rayleigh fading (simplified)
            fading_db = self._calculate_rayleigh_fading()
        else:
            # Rician fading (simplified)
            fading_db = self._calculate_rician_fading(k_factor)
        
        # Cache and return result
        self.fading_cache[cache_key] = fading_db
        return fading_db
    
    def _calculate_rayleigh_fading(self) -> float:
        """
        Calculate Rayleigh fading component.
        
        Returns:
            Fading loss in dB
        """
        # Generate complex Gaussian random variable
        real = self.rng.normal(0, 1/np.sqrt(2))
        imag = self.rng.normal(0, 1/np.sqrt(2))
        
        # Calculate envelope and convert to dB
        envelope = np.sqrt(real**2 + imag**2)
        power_linear = envelope**2
        
        # Convert to dB loss (negative because it's a loss)
        return -10 * np.log10(power_linear)
    
    def _calculate_rician_fading(self, k_factor: float) -> float:
        """
        Calculate Rician fading component.
        
        Args:
            k_factor: Rician K-factor
            
        Returns:
            Fading loss in dB
        """
        # Generate scattered component (complex Gaussian)
        real = self.rng.normal(0, 1/np.sqrt(2))
        imag = self.rng.normal(0, 1/np.sqrt(2))
        
        # Add dominant component (LOS)
        real += np.sqrt(k_factor)
        
        # Calculate envelope and convert to dB
        envelope = np.sqrt(real**2 + imag**2)
        power_linear = envelope**2
        
        # Convert to dB loss (negative because it's a loss)
        return -10 * np.log10(power_linear)
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible fading.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        self.fading_cache.clear()  # Clear cache when seed changes