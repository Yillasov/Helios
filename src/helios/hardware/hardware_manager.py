"""Hardware integration manager for connecting to external RF equipment."""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from helios.core.data_structures import Signal, Waveform, Position
from helios.hardware.interfaces import IRadioHardwareInterface

logger = logging.getLogger(__name__)

class HardwareManager:
    """
    Central manager for hardware device connections and operations.
    Provides a unified interface to control multiple hardware devices.
    """
    
    def __init__(self):
        """Initialize the hardware manager."""
        self.devices = {}  # Dictionary of connected hardware devices
        self.active_device = None  # Currently active device
    
    def register_device(self, device_id: str, device: IRadioHardwareInterface) -> bool:
        """
        Register a hardware device with the manager.
        
        Args:
            device_id: Unique identifier for the device
            device: Hardware device interface implementation
            
        Returns:
            Success status
        """
        if device_id in self.devices:
            logger.warning(f"Device ID {device_id} already registered")
            return False
            
        self.devices[device_id] = device
        logger.info(f"Registered device: {device_id}")
        
        # Set as active device if it's the first one
        if self.active_device is None:
            self.active_device = device_id
            
        return True
    
    def set_active_device(self, device_id: str) -> bool:
        """
        Set the active device for operations.
        
        Args:
            device_id: ID of device to set as active
            
        Returns:
            Success status
        """
        if device_id not in self.devices:
            logger.error(f"Device ID {device_id} not found")
            return False
            
        self.active_device = device_id
        logger.info(f"Set active device to {device_id}")
        return True
    
    def initialize_device(self, device_id: str, config: Dict[str, Any]) -> bool:
        """
        Initialize a specific hardware device.
        
        Args:
            device_id: ID of device to initialize
            config: Configuration parameters
            
        Returns:
            Success status
        """
        if device_id not in self.devices:
            logger.error(f"Device ID {device_id} not found")
            return False
            
        success = self.devices[device_id].initialize(config)
        if success:
            logger.info(f"Initialized device {device_id}")
        else:
            logger.error(f"Failed to initialize device {device_id}")
        return success
    
    def transmit_signal(self, signal: Signal, device_id: Optional[str] = None) -> bool:
        """
        Transmit a signal using the specified or active device.
        
        Args:
            signal: Signal to transmit
            device_id: Optional device ID (uses active device if None)
            
        Returns:
            Success status
        """
        # Use active device if none specified
        device_id = device_id or self.active_device
        if not device_id or device_id not in self.devices:
            logger.error("No valid device specified for transmission")
            return False
        
        device = self.devices[device_id]
        
        # Configure transmitter parameters
        tx_params = {
            "frequency": signal.waveform.center_frequency,
            "power": signal.power,
            "bandwidth": signal.waveform.bandwidth
        }
        
        # Add modulation parameters if available
        if signal.waveform.modulation_type.name != "NONE":
            # Create a separate modulation config instead of adding to tx_params
            modulation_config = {
                "modulation_type": signal.waveform.modulation_type.name
            }
            # Add any modulation-specific parameters
            if hasattr(signal.waveform, 'modulation_params'):
                modulation_config.update(signal.waveform.modulation_params)
            
            # Set modulation configuration separately
            if not device.set_modulation_config(modulation_config):
                logger.error("Failed to set modulation parameters")
                return False
        
        # Set transmitter parameters
        if not device.set_tx_parameters(tx_params):
            logger.error("Failed to set transmitter parameters")
            return False
        
        # Generate samples (simplified - in a real implementation, 
        # this would use the waveform generator)
        sample_rate = 2 * signal.waveform.bandwidth  # Nyquist rate
        duration = signal.waveform.duration or 0.1  # Default 100ms if not specified
        num_samples = int(sample_rate * duration)
        
        # Simple sine wave as placeholder
        t = np.linspace(0, duration, num_samples, endpoint=False)
        samples = np.exp(2j * np.pi * 1000 * t)  # 1 kHz tone
        
        # Transmit the samples
        success = device.transmit_samples(
            samples=samples,
            center_freq=signal.waveform.center_frequency,
            sample_rate=sample_rate,
            gain=10.0  # Default gain
        )
        
        if success:
            logger.info(f"Transmitted signal {signal.id} using device {device_id}")
        else:
            logger.error(f"Failed to transmit signal {signal.id}")
        
        return success
    
    def receive_spectrum(self, 
                        center_freq: float, 
                        bandwidth: float, 
                        duration: float = 0.1,
                        device_id: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Receive spectrum data from the specified or active device.
        
        Args:
            center_freq: Center frequency in Hz
            bandwidth: Bandwidth to capture in Hz
            duration: Duration to capture in seconds
            device_id: Optional device ID (uses active device if None)
            
        Returns:
            Tuple of (frequency_bins, power_spectrum, metadata)
        """
        # Use active device if none specified
        device_id = device_id or self.active_device
        if not device_id or device_id not in self.devices:
            logger.error("No valid device specified for reception")
            return np.array([]), {}
        
        device = self.devices[device_id]
        
        # Configure receiver parameters
        rx_params = {
            "frequency": center_freq,
            "bandwidth": bandwidth,
            "gain": 20.0  # Default gain
        }
        
        # Set receiver parameters
        if not device.set_rx_parameters(rx_params):
            logger.error("Failed to set receiver parameters")
            return np.array([]), {}
        
        # Calculate number of samples based on bandwidth and duration
        sample_rate = 2 * bandwidth  # Nyquist rate
        num_samples = int(sample_rate * duration)
        
        # Receive samples
        samples, metadata = device.receive_samples(
            num_samples=num_samples,
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=20.0  # Default gain
        )
        
        logger.info(f"Received {len(samples)} samples from device {device_id}")
        
        return samples, metadata
    
    def close_all_devices(self) -> None:
        """Close all connected hardware devices."""
        for device_id, device in self.devices.items():
            try:
                device.close()
                logger.info(f"Closed device {device_id}")
            except Exception as e:
                logger.error(f"Error closing device {device_id}: {e}")
        
        self.devices = {}
        self.active_device = None