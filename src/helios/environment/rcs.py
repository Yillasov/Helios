"""Radar Cross Section (RCS) models for various objects."""

import numpy as np
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Union, Callable

from helios.core.data_structures import Position, Orientation
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class RCSModelType(Enum):
    """Types of RCS models."""
    ISOTROPIC = auto()
    SPHERE = auto()
    FLAT_PLATE = auto()
    CYLINDER = auto()
    CORNER_REFLECTOR = auto()
    COMPLEX = auto()

class RCSModel:
    """Base class for RCS models."""
    
    def __init__(self, model_type: RCSModelType, base_rcs: float = 1.0):
        """
        Initialize the RCS model.
        
        Args:
            model_type: Type of RCS model
            base_rcs: Base RCS value in m²
        """
        self.model_type = model_type
        self.base_rcs = base_rcs
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate RCS for given parameters.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            
        Returns:
            RCS value in m²
        """
        # Base implementation returns constant RCS
        return self.base_rcs

class IsotropicRCSModel(RCSModel):
    """
    Isotropic RCS model that returns the same RCS value regardless of angle.
    Useful for simple simulations or as a baseline.
    """
    
    def __init__(self, rcs: float = 1.0):
        """
        Initialize isotropic RCS model.
        
        Args:
            rcs: Constant RCS value in m²
        """
        super().__init__(RCSModelType.ISOTROPIC, rcs)
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """Calculate RCS (constant for isotropic model)."""
        return self.base_rcs

class SphereRCSModel(RCSModel):
    """
    Sphere RCS model based on sphere radius and frequency.
    RCS varies with frequency but not with angle.
    """
    
    def __init__(self, radius: float):
        """
        Initialize sphere RCS model.
        
        Args:
            radius: Sphere radius in meters
        """
        super().__init__(RCSModelType.SPHERE)
        self.radius = radius
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate sphere RCS.
        
        For spheres, RCS depends on radius and wavelength:
        - Rayleigh region (2πr/λ << 1): σ = π·r²·(2πr/λ)⁴
        - Optical region (2πr/λ >> 1): σ = π·r²
        - Mie region (transition): complex behavior
        """
        wavelength = 3e8 / frequency
        circumference_wavelengths = 2 * np.pi * self.radius / wavelength
        
        # Simplified model
        if circumference_wavelengths < 0.1:  # Rayleigh region
            # RCS proportional to r² * (r/λ)⁴
            return np.pi * self.radius**2 * (2 * np.pi * self.radius / wavelength)**4
        elif circumference_wavelengths > 10:  # Optical region
            # RCS equals geometric cross-section
            return np.pi * self.radius**2
        else:  # Mie region (simplified)
            # Approximate with a smooth transition
            return np.pi * self.radius**2 * min(1.0, circumference_wavelengths**2 / 100)

class FlatPlateRCSModel(RCSModel):
    """
    Flat plate RCS model that varies with incident angle.
    Maximum RCS occurs at normal incidence.
    """
    
    def __init__(self, length: float, width: float):
        """
        Initialize flat plate RCS model.
        
        Args:
            length: Plate length in meters
            width: Plate width in meters
        """
        super().__init__(RCSModelType.FLAT_PLATE)
        self.length = length
        self.width = width
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate flat plate RCS.
        
        RCS depends on plate area, wavelength, and incident angle:
        σ = (4π·A²/λ²)·cos²(θ)
        where A is area, λ is wavelength, θ is incident angle from normal
        """
        wavelength = 3e8 / frequency
        area = self.length * self.width
        
        # Calculate plate normal vector (simplified)
        # Assuming plate normal is aligned with z-axis before rotation
        normal = np.array([0, 0, 1])
        
        # Apply rotation based on orientation (simplified)
        # This is a simplified rotation calculation
        cos_roll = np.cos(target_orientation.roll)
        sin_roll = np.sin(target_orientation.roll)
        cos_pitch = np.cos(target_orientation.pitch)
        sin_pitch = np.sin(target_orientation.pitch)
        cos_yaw = np.cos(target_orientation.yaw)
        sin_yaw = np.sin(target_orientation.yaw)
        
        # Simplified rotation matrix application
        normal = np.array([
            sin_pitch,
            -sin_roll * cos_pitch,
            cos_roll * cos_pitch
        ])
        
        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Calculate incident angle
        incident = np.array(incident_direction)
        cos_theta = abs(np.dot(normal, incident))
        
        # Calculate RCS
        rcs = 4 * np.pi * area**2 / wavelength**2 * cos_theta**4
        
        return rcs

class CornerReflectorRCSModel(RCSModel):
    """
    Corner reflector (trihedral) RCS model.
    Provides high RCS over a wide range of angles.
    """
    
    def __init__(self, edge_length: float):
        """
        Initialize corner reflector RCS model.
        
        Args:
            edge_length: Length of each edge in meters
        """
        super().__init__(RCSModelType.CORNER_REFLECTOR)
        self.edge_length = edge_length
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate corner reflector RCS.
        
        For a trihedral corner reflector:
        σ = (4π/3)·(a⁴/λ²)
        where a is edge length and λ is wavelength
        """
        wavelength = 3e8 / frequency
        
        # Basic formula for trihedral corner reflector
        rcs = (4 * np.pi / 3) * (self.edge_length**4 / wavelength**2)
        
        # Apply angular dependence (simplified)
        # Corner reflectors have good performance over wide angles
        incident = np.array(incident_direction)
        
        # Calculate direction to corner (simplified)
        corner_dir = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized
        
        # Apply rotation based on orientation (simplified)
        # This is a very simplified calculation
        cos_angle = abs(np.dot(corner_dir, incident))
        
        # Angular dependence (simplified)
        angular_factor = cos_angle**2
        
        return rcs * angular_factor