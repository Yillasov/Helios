"""Hardware integration interfaces for Helios RF systems."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from helios.core.data_structures import Signal, Waveform, Position


class IRadioHardwareInterface(ABC):
    """Interface for hardware radio integration."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize hardware connection with configuration parameters.
        
        Args:
            config: Hardware-specific configuration parameters
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def transmit_samples(self, 
                        samples: np.ndarray, 
                        center_freq: float,
                        sample_rate: float,
                        gain: float) -> bool:
        """Transmit IQ samples through hardware.
        
        Args:
            samples: Complex IQ samples to transmit
            center_freq: Center frequency in Hz
            sample_rate: Sample rate in Hz
            gain: Transmit gain in dB
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def receive_samples(self, 
                       num_samples: int,
                       center_freq: float,
                       sample_rate: float,
                       gain: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Receive IQ samples from hardware.
        
        Args:
            num_samples: Number of samples to receive
            center_freq: Center frequency in Hz
            sample_rate: Sample rate in Hz
            gain: Receive gain in dB
            
        Returns:
            Tuple of (samples, metadata)
        """
        pass
    
    @abstractmethod
    def set_rx_parameters(self, params: Dict[str, Any]) -> bool:
        """Configure receiver parameters.
        
        Args:
            params: Parameter dictionary (frequency, bandwidth, gain, etc.)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def set_tx_parameters(self, params: Dict[str, Any]) -> bool:
        """Configure transmitter parameters.
        
        Args:
            params: Parameter dictionary (frequency, bandwidth, gain, etc.)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware status.
        
        Returns:
            Status dictionary (temperature, lock status, errors, etc.)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close hardware connection and release resources."""
        pass


class IDigitalDataFormat:
    """Standard digital data format for hardware interfaces."""
    
    COMPLEX_FLOAT32 = "cf32"  # Complex float32 (8 bytes per sample)
    COMPLEX_INT16 = "ci16"    # Complex int16 (4 bytes per sample)
    COMPLEX_INT8 = "ci8"      # Complex int8 (2 bytes per sample)
    REAL_FLOAT32 = "f32"      # Real float32 (4 bytes per sample)
    REAL_INT16 = "i16"        # Real int16 (2 bytes per sample)
    REAL_INT8 = "i8"          # Real int8 (1 byte per sample)