"""Target acquisition probability calculators for RF-based systems."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from helios.core.data_structures import Position, Signal, EnvironmentParameters
from helios.utils.logger import get_logger
from helios.environment.rcs import RCSModel
from helios.environment.propagation import LOSPropagationModel
from helios.core.interfaces import IPropagationModel  # Import the interface instead
from helios.armament.targeting import TargetDesignation

logger = get_logger(__name__)

class AcquisitionMode(Enum):
    """Target acquisition modes."""
    SEARCH = auto()         # Initial search mode
    TRACK = auto()          # Tracking an acquired target
    LOCK = auto()           # Locked onto target
    REACQUISITION = auto()  # Attempting to reacquire a lost target


@dataclass
class AcquisitionParameters:
    """Parameters affecting target acquisition probability."""
    snr_threshold: float = 13.0  # dB
    false_alarm_rate: float = 1e-6  # Probability
    integration_gain: float = 0.0  # dB
    min_detection_range: float = 100.0  # meters
    max_detection_range: float = 50000.0  # meters
    scan_time: float = 10.0  # seconds for full scan
    beam_width: float = 3.0  # degrees
    frequency: float = 10e9  # Hz (X-band)
    transmit_power: float = 1000.0  # Watts
    antenna_gain: float = 30.0  # dBi
    system_losses: float = 3.0  # dB
    
    # Environmental factors
    rain_rate: float = 0.0  # mm/hour
    humidity: float = 0.5  # 0-1 scale
    temperature: float = 293.0  # Kelvin


class TargetAcquisitionCalculator:
    """Calculator for target acquisition probabilities."""
    
    def __init__(self, 
                propagation_model: Optional[IPropagationModel] = None,
                rcs_model: Optional[RCSModel] = None):
        """Initialize the target acquisition calculator.
        
        Args:
            propagation_model: RF propagation model
            rcs_model: Radar cross-section model
        """
        self.propagation_model = propagation_model or LOSPropagationModel()
        self.rcs_model = rcs_model
        self.params = AcquisitionParameters()
        
    def set_parameters(self, params: AcquisitionParameters):
        """Set acquisition parameters.
        
        Args:
            params: Acquisition parameters
        """
        self.params = params
        
    def calculate_single_look_probability(self, 
                                         target: TargetDesignation,
                                         sensor_position: Position,
                                         target_rcs: float,
                                         environment: Optional[EnvironmentParameters] = None) -> float:
        """Calculate probability of detection in a single radar look.
        
        Args:
            target: Target designation
            sensor_position: Position of the sensor
            target_rcs: Target radar cross-section in m²
            environment: Optional environment parameters
            
        Returns:
            Probability of detection (0-1)
        """
        # Calculate distance to target
        target_pos = Position(*target.position)
        distance = sensor_position.distance_to(target_pos)
        
        # Check if target is within detection range
        if distance < self.params.min_detection_range or distance > self.params.max_detection_range:
            return 0.0
        
        # Calculate SNR using radar equation
        # SNR = (P * G² * λ² * σ) / ((4π)³ * R⁴ * k * T * B * L)
        # We'll use a simplified version
        
        wavelength = 3e8 / self.params.frequency
        
        # Convert antenna gain from dBi to ratio
        gain_ratio = 10 ** (self.params.antenna_gain / 10)
        
        # Calculate basic SNR (without environmental factors)
        snr = (self.params.transmit_power * gain_ratio**2 * wavelength**2 * target_rcs) / \
              ((4 * np.pi)**3 * distance**4 * 1.38e-23 * 290 * 1e6)
        
        # Convert to dB
        snr_db = 10 * np.log10(snr)
        
        # Apply system losses
        snr_db -= self.params.system_losses
        
        # Apply integration gain
        snr_db += self.params.integration_gain
        
        # Apply environmental factors if available
        if environment:
            # Simple atmospheric attenuation model
            if environment.temperature > 0:
                temp_factor = environment.temperature / 293.0  # Normalize to 20°C
                snr_db -= 0.5 * (temp_factor - 1) * distance / 1000  # 0.5dB per km per temp factor
            
            # Rain attenuation (simplified)
            if self.params.rain_rate > 0:
                # Higher frequencies affected more by rain
                freq_factor = min(5.0, self.params.frequency / 10e9)
                rain_atten = 0.01 * self.params.rain_rate * freq_factor * distance / 1000
                snr_db -= rain_atten
        
        # Calculate Pd using simplified Marcum Q-function approximation
        # For SNR >> threshold, Pd approaches 1
        # For SNR << threshold, Pd approaches 0
        # Near threshold, we use a sigmoid function
        
        snr_margin = snr_db - self.params.snr_threshold
        if snr_margin <= -10:
            pd = 0.0
        elif snr_margin >= 10:
            pd = 1.0
        else:
            # Sigmoid function centered at threshold
            pd = 1.0 / (1.0 + np.exp(-snr_margin))
        
        logger.debug(f"Single-look Pd={pd:.3f} at range={distance:.1f}m, SNR={snr_db:.1f}dB")
        return pd
    
    def calculate_cumulative_probability(self,
                                        target: TargetDesignation,
                                        sensor_position: Position,
                                        target_rcs: float,
                                        num_scans: int = 1,
                                        environment: Optional[EnvironmentParameters] = None) -> float:
        """Calculate cumulative probability of detection over multiple scans.
        
        Args:
            target: Target designation
            sensor_position: Position of the sensor
            target_rcs: Target radar cross-section in m²
            num_scans: Number of radar scans
            environment: Optional environment parameters
            
        Returns:
            Cumulative probability of detection (0-1)
        """
        # Calculate single-look probability
        p_single = self.calculate_single_look_probability(
            target, sensor_position, target_rcs, environment
        )
        
        # Calculate cumulative probability
        # P_cumulative = 1 - (1 - P_single)^num_scans
        p_cumulative = 1.0 - (1.0 - p_single) ** num_scans
        
        logger.info(f"Cumulative Pd={p_cumulative:.3f} after {num_scans} scans")
        return p_cumulative
    
    def calculate_time_to_acquire(self,
                                 target: TargetDesignation,
                                 sensor_position: Position,
                                 target_rcs: float,
                                 confidence: float = 0.9,
                                 environment: Optional[EnvironmentParameters] = None) -> float:
        """Calculate expected time to acquire target with given confidence.
        
        Args:
            target: Target designation
            sensor_position: Position of the sensor
            target_rcs: Target radar cross-section in m²
            confidence: Required confidence level (0-1)
            environment: Optional environment parameters
            
        Returns:
            Expected time to acquire in seconds
        """
        # Calculate single-look probability
        p_single = self.calculate_single_look_probability(
            target, sensor_position, target_rcs, environment
        )
        
        if p_single <= 0:
            return float('inf')  # Cannot acquire
            
        # Calculate number of scans needed for desired confidence
        # n = log(1-confidence) / log(1-p_single)
        if p_single >= 0.9999:
            num_scans = 1  # Immediate acquisition
        else:
            num_scans = np.log(1 - confidence) / np.log(1 - p_single)
            
        # Calculate time based on scan time
        time_to_acquire = num_scans * self.params.scan_time
        
        logger.info(f"Time to acquire with {confidence:.1%} confidence: {time_to_acquire:.1f}s")
        return time_to_acquire
    
    def calculate_acquisition_envelope(self,
                                      target_rcs: float,
                                      confidence: float = 0.9,
                                      range_steps: int = 20,
                                      environment: Optional[EnvironmentParameters] = None) -> List[Tuple[float, float]]:
        """Calculate acquisition envelope (range vs. time).
        
        Args:
            target_rcs: Target radar cross-section in m²
            confidence: Required confidence level (0-1)
            range_steps: Number of range steps to calculate
            environment: Optional environment parameters
            
        Returns:
            List of (range, time_to_acquire) tuples
        """
        results = []
        
        # Create dummy target and sensor at origin
        target = TargetDesignation("dummy", (0, 0, 0))
        sensor_position = Position(0, 0, 0)
        
        # Calculate for different ranges
        min_range = self.params.min_detection_range
        max_range = self.params.max_detection_range
        
        for i in range(range_steps):
            # Calculate range for this step
            range_factor = i / (range_steps - 1)
            current_range = min_range + range_factor * (max_range - min_range)
            
            # Update target position
            target.position = (current_range, 0, 0)
            
            # Calculate time to acquire
            time_to_acquire = self.calculate_time_to_acquire(
                target, sensor_position, target_rcs, confidence, environment
            )
            
            results.append((current_range, time_to_acquire))
            
        return results