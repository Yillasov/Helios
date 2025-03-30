"""Terrain-based propagation models for tactical environments."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from helios.core.data_structures import Position
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TerrainPoint:
    """Represents a point in the terrain with elevation and material properties."""
    x: float
    y: float
    elevation: float  # meters above sea level
    material: str = "soil"  # soil, water, urban, forest, rock

@dataclass
class TerrainProfile:
    """Represents a terrain profile between two points."""
    points: List[TerrainPoint]
    distance: float  # Total distance in meters
    
    def get_elevation_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return distance and elevation arrays for plotting."""
        distances = np.array([i * self.distance / (len(self.points) - 1) for i in range(len(self.points))])
        elevations = np.array([point.elevation for point in self.points])
        return distances, elevations

class TacticalTerrainModel:
    """Terrain model for tactical environments with RF propagation effects."""
    
    def __init__(self):
        """Initialize the tactical terrain model."""
        self.terrain_data: Dict[Tuple[int, int], TerrainPoint] = {}
        self.material_properties = {
            'soil': {'permittivity': 15.0, 'conductivity': 0.005, 'roughness': 0.1},
            'water': {'permittivity': 80.0, 'conductivity': 0.01, 'roughness': 0.01},
            'rock': {'permittivity': 5.0, 'conductivity': 0.001, 'roughness': 0.3},
            'urban': {'permittivity': 3.0, 'conductivity': 0.01, 'roughness': 0.5},
            'forest': {'permittivity': 13.0, 'conductivity': 0.003, 'roughness': 0.4},
        }
        
    def add_terrain_point(self, x: float, y: float, elevation: float, material: str = "soil"):
        """Add a terrain point to the model."""
        # Round to nearest grid point for simplicity
        grid_x, grid_y = int(round(x)), int(round(y))
        self.terrain_data[(grid_x, grid_y)] = TerrainPoint(x, y, elevation, material)
        
    def get_terrain_profile(self, start_pos: Position, end_pos: Position, samples: int = 50) -> TerrainProfile:
        """Get terrain profile between two positions."""
        # Generate points along the line
        points = []
        for i in range(samples):
            t = i / (samples - 1)
            x = start_pos.x + t * (end_pos.x - start_pos.x)
            y = start_pos.y + t * (end_pos.y - start_pos.y)
            
            # Find nearest terrain point or interpolate
            grid_x, grid_y = int(round(x)), int(round(y))
            if (grid_x, grid_y) in self.terrain_data:
                terrain_point = self.terrain_data[(grid_x, grid_y)]
                points.append(TerrainPoint(x, y, terrain_point.elevation, terrain_point.material))
            else:
                # Simple interpolation from nearby points
                elevation = self._interpolate_elevation(x, y)
                material = self._get_nearest_material(x, y)
                points.append(TerrainPoint(x, y, elevation, material))
        
        distance = start_pos.distance_to(end_pos)
        return TerrainProfile(points, distance)
    
    def _interpolate_elevation(self, x: float, y: float) -> float:
        """Interpolate elevation at a point from nearby known points."""
        # Simple inverse distance weighted interpolation
        grid_x, grid_y = int(round(x)), int(round(y))
        
        # Search nearby grid points
        nearby_points = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (grid_x + dx, grid_y + dy) in self.terrain_data:
                    nearby_points.append(self.terrain_data[(grid_x + dx, grid_y + dy)])
        
        if not nearby_points:
            return 0.0  # Default elevation if no nearby points
        
        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for point in nearby_points:
            dist = np.sqrt((x - point.x)**2 + (y - point.y)**2)
            if dist < 0.001:  # Very close to a known point
                return point.elevation
            
            weight = 1.0 / (dist**2)
            weighted_sum += weight * point.elevation
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_nearest_material(self, x: float, y: float) -> str:
        """Get material type of the nearest known terrain point."""
        grid_x, grid_y = int(round(x)), int(round(y))
        
        # Find nearest point
        min_dist = float('inf')
        nearest_material = "soil"  # Default
        
        for (px, py), point in self.terrain_data.items():
            dist = np.sqrt((px - grid_x)**2 + (py - grid_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_material = point.material
        
        return nearest_material
    
    def calculate_terrain_effects(self, 
                                 start_pos: Position, 
                                 end_pos: Position, 
                                 frequency_hz: float) -> Dict[str, float]:
        """
        Calculate terrain effects on RF propagation.
        
        Args:
            start_pos: Transmitter position
            end_pos: Receiver position
            frequency_hz: Signal frequency in Hz
            
        Returns:
            Dictionary with propagation effects
        """
        # Get terrain profile
        profile = self.get_terrain_profile(start_pos, end_pos)
        
        # Calculate line-of-sight visibility
        los_blocked = self._check_los_blocked(profile, start_pos.z, end_pos.z)
        
        # Calculate additional path loss due to terrain
        terrain_loss = self._calculate_terrain_loss(profile, frequency_hz, start_pos.z, end_pos.z)
        
        # Calculate multipath effects
        multipath_fading = self._calculate_multipath(profile, frequency_hz)
        
        # Calculate diffraction effects for non-LOS paths
        diffraction_gain = 0.0
        if los_blocked:
            diffraction_gain = self._calculate_diffraction(profile, frequency_hz, start_pos.z, end_pos.z)
        
        return {
            "los_blocked": los_blocked,
            "terrain_loss_db": terrain_loss,
            "multipath_fading_db": multipath_fading,
            "diffraction_gain_db": diffraction_gain,
            "total_effect_db": -terrain_loss + diffraction_gain - multipath_fading
        }
    
    def _check_los_blocked(self, profile: TerrainProfile, start_height: float, end_height: float) -> bool:
        """Check if line-of-sight is blocked by terrain."""
        distances, elevations = profile.get_elevation_profile()
        
        # Add heights of start and end points
        elevations[0] += start_height
        elevations[-1] += end_height
        
        # Check if any terrain point blocks the line of sight
        for i in range(1, len(distances) - 1):
            # Calculate height of LOS at this distance
            t = distances[i] / distances[-1]
            los_height = elevations[0] + t * (elevations[-1] - elevations[0])
            
            # If terrain is higher than LOS, it's blocked
            if elevations[i] > los_height:
                return True
        
        return False
    
    def _calculate_terrain_loss(self, profile: TerrainProfile, frequency_hz: float, 
                              start_height: float, end_height: float) -> float:
        """Calculate additional path loss due to terrain."""
        # Basic implementation using ITU terrain model
        # This is a simplified version of the ITU-R P.1546 model
        
        wavelength = 3e8 / frequency_hz
        distance = profile.distance
        
        # Get terrain irregularity parameter
        distances, elevations = profile.get_elevation_profile()
        delta_h = np.std(elevations)  # Standard deviation of terrain heights
        
        # Calculate terrain roughness factor
        roughness = 0.0
        for point in profile.points:
            material_props = self.material_properties.get(point.material, 
                                                        self.material_properties['soil'])
            roughness += material_props['roughness']
        roughness /= len(profile.points)
        
        # Calculate additional loss
        # This is a simplified formula based on empirical models
        terrain_loss = 6.0 * np.log10(frequency_hz / 1e6) + 10.0 * np.log10(delta_h + 1) + 15.0 * roughness
        
        # Adjust for heights
        h_factor = 10 * np.log10((start_height + 1) * (end_height + 1))
        terrain_loss -= h_factor
        
        return max(0, terrain_loss)  # Cannot have negative loss
    
    def _calculate_multipath(self, profile: TerrainProfile, frequency_hz: float) -> float:
        """Calculate multipath fading effects."""
        # Simple model based on terrain roughness and reflectivity
        wavelength = 3e8 / frequency_hz
        
        # Calculate average reflectivity
        reflectivity = 0.0
        for point in profile.points:
            material_props = self.material_properties.get(point.material, 
                                                        self.material_properties['soil'])
            # Simplified reflectivity calculation
            eps_r = material_props['permittivity']
            reflectivity += (eps_r - 1) / (eps_r + 1)
        
        reflectivity /= len(profile.points)
        
        # Simplified multipath calculation
        multipath_fading = 6.0 * reflectivity * (frequency_hz / 1e9)
        
        return multipath_fading
    
    def _calculate_diffraction(self, profile: TerrainProfile, frequency_hz: float, 
                             start_height: float, end_height: float) -> float:
        """Calculate diffraction gain for non-LOS paths."""
        # Simplified knife-edge diffraction model
        wavelength = 3e8 / frequency_hz
        
        distances, elevations = profile.get_elevation_profile()
        
        # Add heights of start and end points
        elevations[0] += start_height
        elevations[-1] += end_height
        
        # Find the most significant obstacle
        max_obstacle_idx = 0
        max_obstacle_height = -float('inf')
        
        for i in range(1, len(distances) - 1):
            # Calculate height of LOS at this distance
            t = distances[i] / distances[-1]
            los_height = elevations[0] + t * (elevations[-1] - elevations[0])
            
            # Calculate obstacle height relative to LOS
            obstacle_height = elevations[i] - los_height
            
            if obstacle_height > max_obstacle_height:
                max_obstacle_height = obstacle_height
                max_obstacle_idx = i
        
        # If no obstacle, return 0 gain
        if max_obstacle_height <= 0:
            return 0.0
        
        # Calculate Fresnel-Kirchhoff diffraction parameter
        d1 = distances[max_obstacle_idx]
        d2 = distances[-1] - d1
        
        # Fresnel parameter v
        v = max_obstacle_height * np.sqrt(2 * (d1 + d2) / (wavelength * d1 * d2))
        
        # Simplified diffraction loss calculation
        if v <= -0.7:
            diffraction_loss = 0
        elif v <= 2.4:
            diffraction_loss = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
        else:
            diffraction_loss = 13 + 20 * np.log10(v)
        
        # Return negative loss as gain
        return -diffraction_loss