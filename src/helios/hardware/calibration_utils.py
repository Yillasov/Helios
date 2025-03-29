"""Calibration utilities for hardware characterization in Helios."""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import json

from helios.hardware.interfaces import IRadioHardwareInterface
from helios.hardware.sdr_factory import SDRFactory

logger = logging.getLogger(__name__)

class CalibrationProfile:
    """Stores calibration data for a specific hardware device."""
    
    def __init__(self, device_id: str, device_type: str):
        """Initialize a calibration profile."""
        self.device_id = device_id
        self.device_type = device_type
        self.timestamp = time.time()
        self.frequency_response = {}  # Frequency-dependent gain/phase corrections
        self.iq_imbalance = {}        # I/Q imbalance at different frequencies
        self.dc_offset = {}           # DC offset at different gain settings
        self.phase_offset = {}        # Phase offset at different frequencies
        self.noise_floor = {}         # Noise floor at different frequencies/gains
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "timestamp": self.timestamp,
            "frequency_response": self.frequency_response,
            "iq_imbalance": self.iq_imbalance,
            "dc_offset": self.dc_offset,
            "phase_offset": self.phase_offset,
            "noise_floor": self.noise_floor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationProfile':
        """Create profile from dictionary."""
        profile = cls(data["device_id"], data["device_type"])
        profile.timestamp = data["timestamp"]
        profile.frequency_response = data["frequency_response"]
        profile.iq_imbalance = data["iq_imbalance"]
        profile.dc_offset = data["dc_offset"]
        profile.phase_offset = data["phase_offset"]
        profile.noise_floor = data["noise_floor"]
        return profile


class HardwareCalibration:
    """Utilities for calibrating and characterizing RF hardware."""
    
    def __init__(self, calibration_dir: Optional[str] = None):
        """Initialize the calibration utility."""
        if calibration_dir is None:
            # Default to a calibration directory in the user's home
            self.calibration_dir = Path.home() / ".helios" / "calibration"
        else:
            self.calibration_dir = Path(calibration_dir)
            
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using calibration directory: {self.calibration_dir}")
    
    def measure_noise_floor(self, device: IRadioHardwareInterface, 
                           frequencies: List[float], 
                           gains: List[float]) -> Dict[str, Any]:
        """
        Measure the noise floor at different frequencies and gain settings.
        
        Args:
            device: Radio hardware interface
            frequencies: List of frequencies to test (Hz)
            gains: List of gain settings to test (dB)
            
        Returns:
            Dictionary of noise floor measurements
        """
        # Check if device is initialized by getting its status
        status = device.get_hardware_status()
        if "error" in status and "not initialized" in status["error"].lower():
            logger.error("Device not initialized")
            return {}
            
        results = {}
        
        for freq in frequencies:
            freq_key = f"{freq/1e6:.1f}MHz"
            results[freq_key] = {}
            
            for gain in gains:
                logger.info(f"Measuring noise floor at {freq/1e6:.1f} MHz with {gain} dB gain")
                
                # Configure receiver
                device.set_rx_parameters({
                    "frequency": freq,
                    "gain": gain,
                    "sample_rate": 2e6  # 2 MHz sample rate
                })
                
                # Allow AGC to settle
                time.sleep(0.5)
                
                # Receive samples
                samples, _ = device.receive_samples(10000, freq, 2e6, gain)
                
                # Calculate power
                power_db = 10 * np.log10(np.mean(np.abs(samples)**2))
                
                results[freq_key][f"{gain}dB"] = power_db
                logger.info(f"Measured noise floor: {power_db:.2f} dB")
                
        return results
    
    def measure_iq_imbalance(self, device: IRadioHardwareInterface,
                            frequencies: List[float]) -> Dict[str, Any]:
        """
        Measure I/Q imbalance at different frequencies.
        
        Args:
            device: Radio hardware interface
            frequencies: List of frequencies to test (Hz)
            
        Returns:
            Dictionary of I/Q imbalance measurements
        """
        # Check if device is initialized by getting its status
        status = device.get_hardware_status()
        if "error" in status and "not initialized" in status["error"].lower():
            logger.error("Device not initialized")
            return {}
            
        results = {}
        
        for freq in frequencies:
            freq_key = f"{freq/1e6:.1f}MHz"
            logger.info(f"Measuring I/Q imbalance at {freq/1e6:.1f} MHz")
            
            # Configure receiver
            device.set_rx_parameters({
                "frequency": freq,
                "gain": 30,  # Mid-range gain
                "sample_rate": 2e6  # 2 MHz sample rate
            })
            
            # Allow settings to settle
            time.sleep(0.5)
            
            # Receive samples
            samples, _ = device.receive_samples(100000, freq, 2e6, 30)
            
            # Calculate I/Q imbalance
            i_samples = np.real(samples)
            q_samples = np.imag(samples)
            
            i_power = np.mean(i_samples**2)
            q_power = np.mean(q_samples**2)
            
            # Amplitude imbalance in dB
            amp_imbalance = 10 * np.log10(i_power / q_power) if q_power > 0 else 0
            
            # Phase imbalance
            iq_correlation = np.mean(i_samples * q_samples)
            phase_imbalance = np.arcsin(iq_correlation / np.sqrt(i_power * q_power)) * 180 / np.pi
            
            results[freq_key] = {
                "amplitude_imbalance_db": amp_imbalance,
                "phase_imbalance_degrees": phase_imbalance
            }
            
            logger.info(f"I/Q imbalance at {freq/1e6:.1f} MHz: "
                       f"Amplitude={amp_imbalance:.2f}dB, Phase={phase_imbalance:.2f}°")
                
        return results
    
    def calibrate_device(self, device_id: str, device: IRadioHardwareInterface) -> CalibrationProfile:
        """
        Perform comprehensive calibration of a device.
        
        Args:
            device_id: Unique identifier for the device
            device: Radio hardware interface
            
        Returns:
            Calibration profile
        """
        # Get device status to check initialization and get device type
        status = device.get_hardware_status()
        if "error" in status and "not initialized" in status["error"].lower():
            logger.error(f"Cannot calibrate device {device_id}: not initialized")
            # Return an empty profile
            return CalibrationProfile(device_id, "unknown")
        
        device_type = status.get("device_type", status.get("board_id", "unknown"))
        
        profile = CalibrationProfile(device_id, device_type)
        
        # Define test frequencies (100 MHz to 1 GHz in steps)
        test_frequencies = [100e6, 200e6, 500e6, 700e6, 900e6]
        
        # Define test gain settings
        test_gains = [10, 20, 30, 40, 50]
        
        # Measure noise floor
        profile.noise_floor = self.measure_noise_floor(device, test_frequencies, [float(gain) for gain in test_gains])
        
        # Measure I/Q imbalance
        profile.iq_imbalance = self.measure_iq_imbalance(device, test_frequencies)
        
        # Save calibration profile
        self.save_calibration_profile(profile)
        
        return profile
    
    def save_calibration_profile(self, profile: CalibrationProfile) -> str:
        """
        Save calibration profile to disk.
        
        Args:
            profile: Calibration profile to save
            
        Returns:
            Path to saved profile
        """
        filename = f"{profile.device_id}_{int(profile.timestamp)}.json"
        filepath = self.calibration_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
            
        logger.info(f"Saved calibration profile to {filepath}")
        return str(filepath)
    
    def load_calibration_profile(self, filepath: str) -> Optional[CalibrationProfile]:
        """
        Load calibration profile from disk.
        
        Args:
            filepath: Path to calibration profile
            
        Returns:
            Loaded calibration profile or None if file not found
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            profile = CalibrationProfile.from_dict(data)
            logger.info(f"Loaded calibration profile for {profile.device_id}")
            return profile
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load calibration profile: {e}")
            return None
    
    def get_latest_profile(self, device_id: str) -> Optional[CalibrationProfile]:
        """
        Get the latest calibration profile for a device.
        
        Args:
            device_id: Device ID to find profile for
            
        Returns:
            Latest calibration profile or None if not found
        """
        profiles = list(self.calibration_dir.glob(f"{device_id}_*.json"))
        if not profiles:
            return None
            
        # Sort by timestamp (part of filename)
        latest = sorted(profiles, key=lambda p: int(p.stem.split('_')[1]))[-1]
        return self.load_calibration_profile(str(latest))


def apply_calibration(samples: np.ndarray, profile: CalibrationProfile, 
                     frequency: float, gain: float) -> np.ndarray:
    """
    Apply calibration corrections to received samples.
    
    Args:
        samples: Complex IQ samples
        profile: Calibration profile
        frequency: Center frequency in Hz
        gain: Gain setting in dB
        
    Returns:
        Corrected samples
    """
    # Find closest frequency and gain in profile
    freq_key = f"{frequency/1e6:.1f}MHz"
    gain_key = f"{gain}dB"
    
    # Get closest frequency if exact match not found
    available_freqs = list(profile.iq_imbalance.keys())
    if freq_key not in available_freqs and available_freqs:
        freq_key = min(available_freqs, 
                      key=lambda f: abs(float(f.replace('MHz', '')) - frequency/1e6))
    
    # Apply I/Q correction if available
    if freq_key in profile.iq_imbalance:
        iq_data = profile.iq_imbalance[freq_key]
        
        # Convert from dB to linear scale
        amp_imbalance = 10 ** (iq_data["amplitude_imbalance_db"] / 20)
        phase_imbalance_rad = iq_data["phase_imbalance_degrees"] * np.pi / 180
        
        # Apply correction
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        
        # Correct amplitude imbalance
        q_samples = q_samples * amp_imbalance
        
        # Correct phase imbalance
        q_corrected = q_samples * np.cos(phase_imbalance_rad) - i_samples * np.sin(phase_imbalance_rad)
        
        # Reconstruct complex samples
        samples = i_samples + 1j * q_corrected
    
    return samples


# Simple usage example
def main():
    """Example usage of calibration utilities."""
    # Create calibration utility
    calibration = HardwareCalibration()
    
    # Create and initialize a device
    device = SDRFactory.create_sdr('rtlsdr')
    if device:
        device.initialize({"device_index": 0})
        
        # Calibrate the device
        profile = calibration.calibrate_device("rtlsdr_main", device)
        
        # Later, when receiving samples
        samples, _ = device.receive_samples(10000, 100e6, 2e6, 30)
        
        # Apply calibration
        corrected = apply_calibration(samples, profile, 100e6, 30)
        
        print(f"Applied calibration: original shape={samples.shape}, corrected shape={corrected.shape}")


if __name__ == "__main__":
    main()