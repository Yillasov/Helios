"""Data structures for representing EM enclosures and apertures."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from enum import Enum

from helios.core.data_structures import Position, Orientation
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class ApertureType(Enum):
    """Types of apertures in an enclosure."""
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"
    SLOT = "slot"
    MESH = "mesh"

class MaterialType(Enum):
    """Common enclosure materials."""
    ALUMINUM = "aluminum"
    STEEL = "steel"
    COPPER = "copper"
    PLASTIC = "plastic"
    COMPOSITE = "composite"

@dataclass
class Material:
    """Material properties for EM calculations."""
    type: MaterialType
    conductivity: float  # S/m
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0
    thickness: float = 0.001  # meters
    
    @classmethod
    def create_common(cls, material_type: MaterialType) -> 'Material':
        """Create a material with common predefined properties."""
        if material_type == MaterialType.ALUMINUM:
            return cls(MaterialType.ALUMINUM, 3.8e7, 1.0, 1.0, 0.002)
        elif material_type == MaterialType.STEEL:
            return cls(MaterialType.STEEL, 1.0e6, 1.0, 1000.0, 0.001)
        elif material_type == MaterialType.COPPER:
            return cls(MaterialType.COPPER, 5.8e7, 1.0, 1.0, 0.001)
        elif material_type == MaterialType.PLASTIC:
            return cls(MaterialType.PLASTIC, 1e-14, 3.0, 1.0, 0.003)
        elif material_type == MaterialType.COMPOSITE:
            return cls(MaterialType.COMPOSITE, 1e4, 4.0, 1.0, 0.002)
        else:
            logger.warning(f"Unknown material type: {material_type}, using default")
            return cls(material_type, 1.0, 1.0, 1.0, 0.001)

@dataclass
class Aperture:
    """Base class for apertures in an enclosure."""
    type: ApertureType
    position: Position  # Center position relative to enclosure
    orientation: Orientation = field(default_factory=Orientation)
    
    def get_area(self) -> float:
        """Get the aperture area in square meters."""
        raise NotImplementedError("Subclasses must implement get_area()")
    
    def get_polarizability(self) -> Tuple[float, float, float]:
        """Get electric and magnetic polarizability (αe, αm_parallel, αm_perpendicular)."""
        raise NotImplementedError("Subclasses must implement get_polarizability()")

@dataclass
class CircularAperture(Aperture):
    """Circular aperture in an enclosure."""
    radius: float = field(default=0.0)  # meters
    
    def __post_init__(self):
        self.type = ApertureType.CIRCULAR
    
    def get_area(self) -> float:
        """Get the aperture area in square meters."""
        return np.pi * self.radius**2
    
    def get_polarizability(self) -> Tuple[float, float, float]:
        """
        Get electric and magnetic polarizability based on Bethe's theory.
        Returns (αe, αm_parallel, αm_perpendicular) in m^3
        """
        # For a circular aperture of radius a:
        # Electric polarizability αe = (2/3) * a^3
        # Magnetic polarizability αm = (4/3) * a^3
        a_cubed = self.radius**3
        alpha_e = (2/3) * a_cubed
        alpha_m_parallel = (4/3) * a_cubed
        alpha_m_perpendicular = (4/3) * a_cubed
        
        return (alpha_e, alpha_m_parallel, alpha_m_perpendicular)

@dataclass
class RectangularAperture(Aperture):
    """Rectangular aperture in an enclosure."""
    width: float = field(default=0.0)  # meters (x-dimension)
    height: float = field(default=0.0)  # meters (y-dimension)
    
    def __post_init__(self):
        self.type = ApertureType.RECTANGULAR
    
    def get_area(self) -> float:
        """Get the aperture area in square meters."""
        return self.width * self.height
    
    def get_polarizability(self) -> Tuple[float, float, float]:
        """
        Get electric and magnetic polarizability for rectangular aperture.
        Returns (αe, αm_parallel, αm_perpendicular) in m^3
        
        Uses approximation formulas for rectangular apertures.
        """
        # For a rectangular aperture with width a and height b (a > b):
        a = max(self.width, self.height)
        b = min(self.width, self.height)
        
        # Electric polarizability
        alpha_e = (np.pi * a * b**2) / 3
        
        # Magnetic polarizability (parallel to long dimension)
        alpha_m_parallel = (np.pi * a**3 * b) / 3
        
        # Magnetic polarizability (parallel to short dimension)
        alpha_m_perpendicular = (np.pi * a * b**3) / 3
        
        return (alpha_e, alpha_m_parallel, alpha_m_perpendicular)

@dataclass
class SlotAperture(Aperture):
    """Slot (narrow rectangular) aperture in an enclosure."""
    length: float = field(default=0.0)  # meters (long dimension)
    width: float = field(default=0.0)   # meters (narrow dimension)
    
    def __post_init__(self):
        self.type = ApertureType.SLOT
    
    def get_area(self) -> float:
        """Get the aperture area in square meters."""
        return self.length * self.width
    
    def get_polarizability(self) -> Tuple[float, float, float]:
        """
        Get electric and magnetic polarizability for slot aperture.
        Returns (αe, αm_parallel, αm_perpendicular) in m^3
        
        Uses approximation for narrow slots where length >> width.
        """
        # For a slot with length L and width w (L >> w):
        L = self.length
        w = self.width
        
        # Electric polarizability (parallel to width)
        alpha_e = (np.pi * w**2 * L) / 12
        
        # Magnetic polarizability (parallel to length)
        alpha_m_parallel = (np.pi * L**3) / 12
        
        # Magnetic polarizability (perpendicular to slot plane)
        alpha_m_perpendicular = (np.pi * w**2 * L) / 6
        
        return (alpha_e, alpha_m_parallel, alpha_m_perpendicular)

@dataclass
class MeshAperture(Aperture):
    """Wire mesh aperture in an enclosure."""
    width: float = field(default=0.0)  # Overall width in meters
    height: float = field(default=0.0)  # Overall height in meters
    wire_spacing: float = field(default=0.0)  # Distance between wires in meters
    wire_diameter: float = field(default=0.0)  # Wire diameter in meters
    
    def __post_init__(self):
        self.type = ApertureType.MESH
    
    def get_area(self) -> float:
        """Get the effective aperture area in square meters."""
        # Calculate open area of the mesh
        cell_area = self.wire_spacing**2
        wire_area = self.wire_diameter * self.wire_spacing * 2 - self.wire_diameter**2
        open_fraction = (cell_area - wire_area) / cell_area
        return self.width * self.height * open_fraction
    
    def get_polarizability(self) -> Tuple[float, float, float]:
        """
        Get effective polarizability for mesh aperture.
        Returns (αe, αm_parallel, αm_perpendicular) in m^3
        
        Uses approximation based on mesh properties.
        """
        # Mesh is modeled as an array of small apertures
        # Effective polarizability is reduced by shielding effectiveness
        
        # Calculate shielding effectiveness (SE) in dB
        wavelength = 0.3 / 2.5e9  # Approximate wavelength at 2.5 GHz (middle of 1-4 GHz)
        if self.wire_spacing < wavelength / 10:
            # For electrically small mesh
            se_db = 20 * np.log10(wavelength / (2 * self.wire_spacing))
        else:
            # For larger mesh
            se_db = 10 * np.log10(1 + (wavelength / (2 * self.wire_spacing))**2)
        
        # Convert SE to linear scale reduction factor
        reduction = 10**(-se_db / 20)
        
        # Calculate effective polarizability as if it were a rectangular aperture
        rect_aperture = RectangularAperture(
            type=ApertureType.RECTANGULAR,
            position=self.position,
            width=self.width,
            height=self.height
        )
        base_polarizability = rect_aperture.get_polarizability()
        
        # Apply reduction factor
        return tuple(p * reduction for p in base_polarizability)

@dataclass
class Enclosure:
    """Represents an EM enclosure with apertures."""
    position: Position  # Center position in global coordinates
    orientation: Orientation = field(default_factory=Orientation)
    dimensions: Tuple[float, float, float] = (0.3, 0.2, 0.1)  # Width, height, depth in meters
    material: Material = field(default_factory=lambda: Material.create_common(MaterialType.ALUMINUM))
    apertures: List[Aperture] = field(default_factory=list)
    
    # Internal points of interest (e.g., sensitive components)
    points_of_interest: Dict[str, Position] = field(default_factory=dict)
    
    def add_aperture(self, aperture: Aperture) -> None:
        """Add an aperture to the enclosure."""
        self.apertures.append(aperture)
    
    def add_point_of_interest(self, name: str, position: Position) -> None:
        """Add an internal point of interest."""
        self.points_of_interest[name] = position
    
    def get_global_position(self, local_position: Position) -> Position:
        """Convert a local position to global coordinates."""
        # Simple implementation without full rotation matrix
        # For a complete implementation, use proper rotation matrices based on orientation
        
        # Apply rotation (simplified - only considering yaw/heading)
        cos_yaw = np.cos(self.orientation.yaw)
        sin_yaw = np.sin(self.orientation.yaw)
        
        rotated_x = local_position.x * cos_yaw - local_position.y * sin_yaw
        rotated_y = local_position.x * sin_yaw + local_position.y * cos_yaw
        rotated_z = local_position.z
        
        # Apply translation
        global_x = rotated_x + self.position.x
        global_y = rotated_y + self.position.y
        global_z = rotated_z + self.position.z
        
        return Position(global_x, global_y, global_z)