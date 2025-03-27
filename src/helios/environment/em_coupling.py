"""Electromagnetic coupling models for enclosures with apertures."""

import numpy as np
# Add 'Any' to the import list
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

from helios.core.data_structures import Position, Signal
from helios.environment.em_structures import (
    Enclosure, Aperture, CircularAperture, RectangularAperture, SlotAperture, MeshAperture
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

# Physical constants
EPSILON_0 = 8.85418782e-12  # Vacuum permittivity (F/m)
MU_0 = 4 * np.pi * 1e-7     # Vacuum permeability (H/m)
SPEED_OF_LIGHT = 299792458  # Speed of light (m/s)

@dataclass
class EMField:
    """Represents an electromagnetic field at a point."""
    e_field: Tuple[float, float, float]  # Electric field vector (V/m)
    h_field: Tuple[float, float, float]  # Magnetic field vector (A/m)
    frequency: float  # Frequency in Hz
    
    @property
    def e_magnitude(self) -> float:
        """Get the magnitude of the electric field."""
        return np.sqrt(sum(e**2 for e in self.e_field))
    
    @property
    def h_magnitude(self) -> float:
        """Get the magnitude of the magnetic field."""
        return np.sqrt(sum(h**2 for h in self.h_field))
    
    @property
    def impedance(self) -> float:
        """Calculate the wave impedance."""
        if self.h_magnitude > 0:
            return self.e_magnitude / self.h_magnitude
        return 0.0
    
    @property
    def power_density(self) -> float:
        """Calculate the power density in W/m²."""
        return self.e_magnitude * self.h_magnitude

@dataclass
class CouplingResult:
    """Results of EM coupling calculation."""
    incident_field: EMField
    coupled_field: Dict[str, EMField]  # Field at each point of interest
    induced_voltage: Dict[str, float]  # Induced voltage at each point of interest
    induced_current: Dict[str, float]  # Induced current at each point of interest
    aperture_transmission: Dict[str, float]  # Transmission coefficient for each aperture

class BetheApertureCoupling:
    """
    Implements Bethe's small aperture coupling theory for EM fields.
    
    This model calculates coupling through electrically small apertures
    (aperture dimensions << wavelength) based on Bethe's theory.
    """
    
    def __init__(self, frequency_range: Tuple[float, float] = (1e9, 4e9)):
        """
        Initialize the Bethe aperture coupling model.
        
        Args:
            frequency_range: Min and max frequency in Hz (default: 1-4 GHz)
        """
        self.min_frequency = frequency_range[0]
        self.max_frequency = frequency_range[1]
        
        # Calculate wavelength range
        self.min_wavelength = SPEED_OF_LIGHT / self.max_frequency
        self.max_wavelength = SPEED_OF_LIGHT / self.min_frequency
    
    def calculate_coupling(
        self, 
        enclosure: Enclosure, 
        incident_field: EMField,
        characteristic_length: float = 0.1  # Characteristic length for voltage calculation
    ) -> CouplingResult:
        """
        Calculate EM coupling through apertures into an enclosure.
        
        Args:
            enclosure: The enclosure with apertures
            incident_field: Incident EM field
            characteristic_length: Characteristic length for voltage calculation (m)
            
        Returns:
            CouplingResult with coupled fields and induced voltages
        """
        # Check if frequency is in valid range
        if not (self.min_frequency <= incident_field.frequency <= self.max_frequency):
            logger.warning(f"Frequency {incident_field.frequency/1e6:.1f} MHz outside valid range "
                          f"({self.min_frequency/1e6:.1f}-{self.max_frequency/1e6:.1f} MHz)")
        
        # Calculate wavelength
        wavelength = SPEED_OF_LIGHT / incident_field.frequency
        
        # Initialize result containers
        coupled_field = {}
        induced_voltage = {}
        induced_current = {}
        aperture_transmission = {}
        
        # Process each aperture
        for i, aperture in enumerate(enclosure.apertures):
            # Check if aperture is electrically small
            max_dimension = self._get_max_dimension(aperture)
            if max_dimension > wavelength / 6:
                logger.warning(f"Aperture {i} may be too large for Bethe's theory: "
                              f"{max_dimension:.2f}m > λ/6 ({wavelength/6:.2f}m)")
            
            # Calculate aperture polarizability
            alpha_e, alpha_m_parallel, alpha_m_perpendicular = aperture.get_polarizability()
            
            # Calculate transmission coefficient (simplified)
            # Transmission is proportional to (aperture_area / wavelength²) for small apertures
            trans_coeff = aperture.get_area() / (wavelength**2)
            aperture_transmission[f"aperture_{i}"] = trans_coeff
            
            # For each point of interest, calculate coupled field
            for name, local_position in enclosure.points_of_interest.items():
                # Calculate vector from aperture to point
                r_vector = self._calculate_vector(aperture.position, local_position)
                r_magnitude = np.sqrt(sum(r**2 for r in r_vector))
                r_unit = tuple(r / r_magnitude for r in r_vector)
                
                # Skip if point is too close to aperture
                if r_magnitude < max_dimension:
                    logger.warning(f"Point {name} too close to aperture {i}, results may be inaccurate")
                    continue
                
                # Calculate coupled fields using Bethe's theory
                e_coupled, h_coupled = self._calculate_coupled_fields(
                    incident_field, alpha_e, alpha_m_parallel, alpha_m_perpendicular,
                    r_vector, r_magnitude, aperture
                )
                
                # Store the coupled field
                point_key = f"{name}_from_aperture_{i}"
                coupled_field[point_key] = EMField(
                    e_field=e_coupled,
                    h_field=h_coupled,
                    frequency=incident_field.frequency
                )
                
                # Calculate induced voltage (E·L approximation)
                e_magnitude = np.sqrt(sum(e**2 for e in e_coupled))
                voltage = e_magnitude * characteristic_length
                induced_voltage[point_key] = voltage
                
                # Calculate induced current (simplified using impedance)
                # Assuming a typical impedance of electronic components
                typical_impedance = 50.0  # ohms
                current = voltage / typical_impedance
                induced_current[point_key] = current
        
        # Combine fields from multiple apertures at each point
        for name in enclosure.points_of_interest:
            # Find all fields for this point
            point_fields = {k: v for k, v in coupled_field.items() if k.startswith(f"{name}_from_aperture_")}
            
            if point_fields:
                # Combine E-fields (vector sum)
                combined_e = [0.0, 0.0, 0.0]
                combined_h = [0.0, 0.0, 0.0]
                
                for field in point_fields.values():
                    for i in range(3):
                        combined_e[i] += field.e_field[i]
                        combined_h[i] += field.h_field[i]
                
                # Store combined field
                coupled_field[name] = EMField(
                    e_field=(combined_e[0], combined_e[1], combined_e[2]),
                    h_field=(combined_h[0], combined_h[1], combined_h[2]),
                    frequency=incident_field.frequency
                )
                
                # Calculate combined induced voltage
                e_magnitude = np.sqrt(sum(e**2 for e in combined_e))
                voltage = e_magnitude * characteristic_length
                induced_voltage[name] = voltage
                
                # Calculate combined induced current
                typical_impedance = 50.0  # ohms
                current = voltage / typical_impedance
                induced_current[name] = current
        
        return CouplingResult(
            incident_field=incident_field,
            coupled_field=coupled_field,
            induced_voltage=induced_voltage,
            induced_current=induced_current,
            aperture_transmission=aperture_transmission
        )
    
    def _get_max_dimension(self, aperture: Aperture) -> float:
        """Get the maximum dimension of an aperture."""
        if isinstance(aperture, CircularAperture):
            return 2 * aperture.radius
        elif isinstance(aperture, RectangularAperture):
            return max(aperture.width, aperture.height)
        elif isinstance(aperture, SlotAperture):
            return aperture.length
        elif isinstance(aperture, MeshAperture):
            return max(aperture.width, aperture.height)
        else:
            # Default case
            return 0.01  # 1cm default
    
    def _calculate_vector(self, start: Position, end: Position) -> Tuple[float, float, float]:
        """Calculate vector from start to end position."""
        return (end.x - start.x, end.y - start.y, end.z - start.z)
    
    def _calculate_coupled_fields(
        self,
        incident_field: EMField,
        alpha_e: float,
        alpha_m_parallel: float,
        alpha_m_perpendicular: float,
        r_vector: Tuple[float, float, float],
        r_magnitude: float,
        aperture: Aperture
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Calculate coupled E and H fields based on Bethe's small aperture theory.
        
        This is a simplified implementation of Bethe's theory for small apertures.
        """
        # Extract incident fields
        E_inc = incident_field.e_field
        H_inc = incident_field.h_field
        
        # Calculate wave impedance of free space
        eta = 120 * np.pi  # ~377 ohms
        
        # Calculate wave number
        k = 2 * np.pi * incident_field.frequency / SPEED_OF_LIGHT
        
        # Unit vector in r direction
        r_unit = tuple(r / r_magnitude for r in r_vector)
        
        # Calculate electric dipole moment (p = ε₀·αₑ·E_inc)
        p = tuple(EPSILON_0 * alpha_e * E for E in E_inc)
        
        # Calculate magnetic dipole moment (m = αₘ·H_inc)
        # This is simplified - in reality, αₘ is a tensor
        m = tuple(alpha_m_parallel * H for H in H_inc)
        
        # Calculate coupled E-field from electric and magnetic dipoles
        # E = [3r(r·p) - p]/r³ * k²/(4πε₀) * exp(-jkr)
        # Simplified without phase factor exp(-jkr)
        
        # Electric dipole contribution to E-field
        p_dot_r = sum(p[i] * r_unit[i] for i in range(3))
        E_p = [0.0, 0.0, 0.0]
        for i in range(3):
            E_p[i] = (3 * r_unit[i] * p_dot_r - p[i]) * k**2 / (4 * np.pi * EPSILON_0 * r_magnitude**3)
        
        # Magnetic dipole contribution to E-field
        # E = -jωμ₀/(4π) * [m×r]/r³ * (1+jkr) * exp(-jkr)
        # Simplified without phase factors
        m_cross_r = (
            m[1] * r_unit[2] - m[2] * r_unit[1],
            m[2] * r_unit[0] - m[0] * r_unit[2],
            m[0] * r_unit[1] - m[1] * r_unit[0]
        )
        
        E_m = [0.0, 0.0, 0.0]
        for i in range(3):
            E_m[i] = 2 * np.pi * incident_field.frequency * MU_0 * m_cross_r[i] / (4 * np.pi * r_magnitude**2)
        
        # Total E-field
        E_coupled = (E_p[0] + E_m[0], E_p[1] + E_m[1], E_p[2] + E_m[2])
        
        # Calculate coupled H-field
        # For simplicity, derive from E using plane wave approximation
        # H = (r × E) / (μ₀·c)
        E_coupled_mag = np.sqrt(sum(E**2 for E in E_coupled))
        H_coupled_mag = E_coupled_mag / eta
        
        # Simplified H-field direction (perpendicular to E and r)
        H_coupled = (
            H_coupled_mag * (r_unit[1] * E_coupled[2] - r_unit[2] * E_coupled[1]),
            H_coupled_mag * (r_unit[2] * E_coupled[0] - r_unit[0] * E_coupled[2]),
            H_coupled_mag * (r_unit[0] * E_coupled[1] - r_unit[1] * E_coupled[0])
        )
        
        # Normalize H-field if needed
        H_mag = np.sqrt(sum(H**2 for H in H_coupled))
        if H_mag > 0:
            H_coupled = tuple(H * H_coupled_mag / H_mag for H in H_coupled)
        
        return E_coupled, H_coupled


class MethodOfMomentsCoupling:
    """
    Placeholder for Method of Moments (MoM) based coupling calculation.
    MoM provides accurate solutions for specific, usually simple, geometries
    but is computationally intensive.
    """
    def __init__(self, geometry_file: Optional[str] = None):
        """
        Initialize the MoM coupling model.
        Args:
            geometry_file: Path to a file defining the structure's geometry.
        """
        self.geometry_file = geometry_file
        logger.info("Initialized MethodOfMomentsCoupling (Placeholder)")
        if geometry_file:
            self._load_geometry(geometry_file)

    def _load_geometry(self, geometry_file: str):
        """Placeholder for loading and meshing geometry."""
        logger.debug(f"Loading geometry from {geometry_file} for MoM (Placeholder)")
        # In a real implementation: Load mesh, define basis functions, etc.
        pass

    def calculate_coupling(
        self,
        enclosure: Enclosure, # Or a more MoM-specific geometry representation
        incident_field: EMField,
    ) -> CouplingResult:
        """
        Placeholder for calculating coupling using MoM.
        Args:
            enclosure: The structure where coupling occurs.
            incident_field: Incident EM field.
        Returns:
            CouplingResult with coupled fields/currents.
        """
        logger.warning("Method of Moments coupling calculation is not implemented yet.")
        # In a real implementation:
        # 1. Define integral equation based on geometry and fields.
        # 2. Discretize using basis functions (MoM).
        # 3. Formulate and solve the matrix equation (Z * I = V).
        # 4. Calculate coupled fields/currents from the solution (I).
        return CouplingResult(
            incident_field=incident_field,
            coupled_field={},
            induced_voltage={}, 
            induced_current={},
            aperture_transmission={}
        )

class RandomCouplingModel:
    """
    Placeholder for Random Coupling Model (RCM) integration.
    RCM is suitable for complex, statistically characterized enclosures
    (electrically large and chaotic cavities).
    """
    def __init__(self, statistical_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the RCM coupling model.
        Args:
            statistical_params: Dictionary of RCM parameters (e.g., loss factor,
                                  aperture characteristics, statistical distributions).
        """
        self.params = statistical_params or {}
        self.radiation_impedance = self.params.get("radiation_impedance", 377.0) # Example param
        self.loss_factor = self.params.get("loss_factor", 0.01) # Example param
        logger.info("Initialized RandomCouplingModel (Placeholder)")

    def calculate_coupling_statistics(
        self,
        enclosure_stats: Dict[str, Any], # Statistical properties of the enclosure
        incident_field_stats: Dict[str, Any], # Statistical properties of the field
        frequency: float
    ) -> Dict[str, Any]:
        """
        Placeholder for calculating statistical coupling properties using RCM.
        Args:
            enclosure_stats: Statistical description of the enclosure.
            incident_field_stats: Statistical description of the incident field.
            frequency: Frequency of interest.
        Returns:
            Dictionary containing statistical measures of coupling (e.g., mean/max field,
            standard deviation).
        """
        logger.warning("Random Coupling Model calculation is not implemented yet.")
        # In a real implementation:
        # 1. Use RCM formulas based on statistical parameters (loss, impedance, etc.).
        # 2. Calculate statistical moments (mean, variance) of quantities like
        #    impedance, power balance, field levels inside the cavity.
        # 3. Return statistical results, not deterministic fields.
        return {"mean_coupled_power_dbm": -50.0, "std_dev_db": 10.0} # Example output

# Potential additions:
# - An interface class (e.g., ICouplingModel) that all models implement.
# - Factory function to select the appropriate model based on configuration or frequency/size regime.