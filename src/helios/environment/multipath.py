"""Multipath propagation models for RF simulation."""

import numpy as np
import math
from scipy.constants import c as SPEED_OF_LIGHT

from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.core.interfaces import IPropagationModel
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class TwoRayGroundModel(IPropagationModel):
    """
    Two-ray ground reflection model that accounts for direct path and ground reflection.
    Provides more accurate path loss prediction than free space model for longer distances.
    """
    
    def __init__(self, ground_dielectric_constant: float = 15.0, ground_conductivity: float = 0.005):
        """
        Initialize the two-ray ground model.
        
        Args:
            ground_dielectric_constant: Relative permittivity of ground (default: 15.0 for moist ground)
            ground_conductivity: Conductivity of ground in S/m (default: 0.005 S/m for moist ground)
        """
        self.ground_dielectric_constant = ground_dielectric_constant
        self.ground_conductivity = ground_conductivity
    
    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss using the two-ray ground reflection model.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB
        """
        # Calculate direct distance
        direct_distance = tx_position.distance_to(rx_position)
        
        if direct_distance <= 1e-6 or frequency <= 0:
            logger.warning(f"Invalid input for two-ray model: distance={direct_distance:.2f}m, freq={frequency:.2f}Hz")
            return float('inf')
        
        # Calculate wavelength
        wavelength = SPEED_OF_LIGHT / frequency
        
        # Calculate heights (assuming z is height above ground)
        ht = tx_position.z
        hr = rx_position.z
        
        # Ensure minimum heights to avoid division by zero
        ht = max(ht, 0.1)  # Minimum 10cm height
        hr = max(hr, 0.1)  # Minimum 10cm height
        
        # Calculate ground distance (horizontal distance)
        dx = np.sqrt((tx_position.x - rx_position.x)**2 + (tx_position.y - rx_position.y)**2)
        
        # Calculate reflection point distance from transmitter
        # For simplicity, assume reflection point is at the midpoint
        d1 = dx / 2
        d2 = dx - d1
        
        # Calculate path lengths
        direct_path = direct_distance
        reflected_path = np.sqrt(d1**2 + (ht)**2) + np.sqrt(d2**2 + (hr)**2)
        
        # Calculate phase difference
        phase_diff = 2 * np.pi * (reflected_path - direct_path) / wavelength
        
        # Calculate reflection coefficient (simplified)
        # Using Fresnel reflection coefficient for vertical polarization
        sin_theta = (ht + hr) / reflected_path
        cos_theta = dx / reflected_path
        
        # Simplified reflection coefficient calculation
        reflection_coef = -1.0  # Simplified: perfect reflection with 180° phase shift
        
        # Calculate received power ratio using two-ray formula
        # |E_r/E_0|^2 = |1 + Γ*exp(jΔφ)|^2
        power_ratio = np.abs(1 + reflection_coef * np.exp(1j * phase_diff))**2
        
        # For large distances, use the approximation: P_r/P_t = (h_t*h_r)^2 / d^4
        if dx > 10 * (ht + hr):  # Far field approximation
            power_ratio = (ht * hr)**2 / dx**4
            path_loss_db = -10 * np.log10(power_ratio)
        else:
            # Convert power ratio to dB
            path_loss_db = -10 * np.log10(power_ratio * (wavelength / (4 * np.pi * direct_distance))**2)
        
        logger.debug(f"Two-ray model: dist={direct_distance:.2f}m, freq={frequency/1e6:.2f}MHz, loss={path_loss_db:.2f}dB")
        
        return max(0.0, path_loss_db)
    
    def apply_propagation_effects(self, 
                                 signal: Signal, 
                                 rx_platform: Platform, 
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply two-ray propagation effects to a signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            environment: Environmental parameters
            
        Returns:
            Modified signal with propagation effects applied
        """
        # Create a copy of the signal to modify
        modified_signal = Signal(
            id=signal.id,
            source_id=signal.source_id,
            waveform=signal.waveform,
            origin=signal.origin,
            source_velocity=signal.source_velocity,
            emission_time=signal.emission_time,
            direction=signal.direction,
            power=signal.power,
            polarization=signal.polarization
        )
        
        # Calculate path loss
        path_loss = self.calculate_path_loss(
            signal.origin, 
            rx_platform.position,
            signal.waveform.center_frequency,
            environment
        )
        
        # Apply path loss to signal power
        modified_signal.power = signal.power - path_loss
        
        # Calculate propagation delay (direct path)
        distance = signal.origin.distance_to(rx_platform.position)
        modified_signal.propagation_delay = distance / SPEED_OF_LIGHT
        
        # Calculate Doppler shift (simplified)
        # Project relative velocity onto the line connecting tx and rx
        tx_to_rx = np.array([
            rx_platform.position.x - signal.origin.x,
            rx_platform.position.y - signal.origin.y,
            rx_platform.position.z - signal.origin.z
        ])
        
        # Normalize
        if distance > 0:
            tx_to_rx = tx_to_rx / distance
        
        # Relative velocity vector
        rel_vel = np.array([
            rx_platform.velocity.x - signal.source_velocity.x,
            rx_platform.velocity.y - signal.source_velocity.y,
            rx_platform.velocity.z - signal.source_velocity.z
        ])
        
        # Radial velocity component
        radial_velocity = np.dot(rel_vel, tx_to_rx)
        
        # Calculate Doppler shift
        modified_signal.doppler_shift = radial_velocity * signal.waveform.center_frequency / SPEED_OF_LIGHT
        
        return modified_signal