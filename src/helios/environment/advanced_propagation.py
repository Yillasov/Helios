"""Advanced RF propagation models including terrain and urban environments."""

import numpy as np
import math
from typing import Dict, Optional, List, Tuple
from scipy.constants import c as SPEED_OF_LIGHT
from dataclasses import dataclass

from helios.core.interfaces import IPropagationModel
from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TerrainProfile:
    """Represents terrain elevation data between two points."""
    distances: np.ndarray  # Distance points along the path (m)
    elevations: np.ndarray  # Elevation at each point (m)
    material_types: List[str]  # Material type at each point (e.g., 'soil', 'water', 'urban')


@dataclass
class BuildingData:
    """Represents building data for urban propagation."""
    position: Position  # Center position
    width: float  # Width in meters
    length: float  # Length in meters
    height: float  # Height in meters
    material: str  # Building material type


class TerrainPropagationModel(IPropagationModel):
    """
    Terrain-based propagation model that accounts for diffraction over terrain features.
    Uses the Longley-Rice Irregular Terrain Model (ITM) approach.
    """
    
    def __init__(self, terrain_resolution: float = 10.0):
        """
        Initialize the terrain propagation model.
        
        Args:
            terrain_resolution: Resolution of terrain sampling in meters
        """
        self.terrain_resolution = terrain_resolution
        self.terrain_data = {}  # Will store terrain data
        self.material_properties = {
            'soil': {'permittivity': 15.0, 'conductivity': 0.005},
            'water': {'permittivity': 80.0, 'conductivity': 0.01},
            'rock': {'permittivity': 5.0, 'conductivity': 0.001},
            'urban': {'permittivity': 3.0, 'conductivity': 0.01},
            'forest': {'permittivity': 13.0, 'conductivity': 0.003},
        }
    
    def load_terrain_data(self, filename: str) -> None:
        """
        Load terrain data from file.
        
        Args:
            filename: Path to terrain data file
        """
        # Implementation would load terrain data from file
        # For now, we'll just log that this would happen
        logger.info(f"Would load terrain data from {filename}")
        
    def get_terrain_profile(self, tx_position: Position, rx_position: Position) -> TerrainProfile:
        """
        Get terrain profile between transmitter and receiver.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            
        Returns:
            Terrain profile between the two points
        """
        # In a real implementation, this would query a terrain database
        # For demonstration, we'll create a synthetic profile
        distance = tx_position.distance_to(rx_position)
        num_points = max(2, int(distance / self.terrain_resolution))
        
        # Create distance points
        distances = np.linspace(0, distance, num_points)
        
        # Create synthetic terrain with some hills
        elevations = np.zeros(num_points)
        for i in range(1, 4):
            # Add some random hills
            hill_pos = distance * (0.2 * i)
            hill_width = distance * 0.1
            hill_height = 20.0 * i
            elevations += hill_height * np.exp(-((distances - hill_pos) / hill_width) ** 2)
        
        # Add tx and rx heights
        elevations[0] = tx_position.z
        elevations[-1] = rx_position.z
        
        # Assign materials (simplified)
        materials = ['soil'] * num_points
        
        return TerrainProfile(distances, elevations, materials)
    
    def calculate_diffraction_loss(self, profile: TerrainProfile, frequency: float) -> float:
        """
        Calculate diffraction loss over terrain using knife-edge diffraction model.
        
        Args:
            profile: Terrain profile
            frequency: Signal frequency in Hz
            
        Returns:
            Diffraction loss in dB
        """
        wavelength = SPEED_OF_LIGHT / frequency
        
        # Find line of sight path
        h_tx = profile.elevations[0]
        h_rx = profile.elevations[-1]
        distance = profile.distances[-1]
        
        # Calculate straight line between tx and rx
        slope = (h_rx - h_tx) / distance
        los_heights = h_tx + slope * profile.distances
        
        # Find the point with maximum obstruction
        clearances = profile.elevations - los_heights
        if np.max(clearances) <= 0:
            # No obstruction
            return 0.0
        
        # Find the point with maximum obstruction
        max_idx = np.argmax(clearances)
        h_obs = clearances[max_idx]
        d1 = profile.distances[max_idx]
        d2 = distance - d1
        
        # Calculate Fresnel-Kirchhoff diffraction parameter
        v = h_obs * np.sqrt(2 * distance / (wavelength * d1 * d2))
        
        # Calculate diffraction loss using approximation
        if v <= -1:
            loss_db = 0
        elif v <= 0:
            loss_db = 20 * np.log10(0.5 - 0.62 * v)
        elif v <= 1:
            loss_db = 20 * np.log10(0.5 * np.exp(-0.95 * v))
        elif v <= 2.4:
            loss_db = 20 * np.log10(0.4 - np.sqrt(0.1184 - (0.38 - 0.1 * v)**2))
        else:
            loss_db = 20 * np.log10(0.225 / v)
        
        return -loss_db  # Convert to positive loss
    
    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss considering terrain effects.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB
        """
        # Get terrain profile
        profile = self.get_terrain_profile(tx_position, rx_position)
        
        # Calculate free space path loss
        distance = tx_position.distance_to(rx_position)
        if distance <= 1e-6 or frequency <= 0:
            return float('inf')
        
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * math.pi / SPEED_OF_LIGHT)
        
        # Calculate diffraction loss
        diffraction_loss = self.calculate_diffraction_loss(profile, frequency)
        
        # Calculate ground reflection effects (simplified)
        ground_loss = 0.0
        for i in range(len(profile.material_types)):
            if profile.material_types[i] in self.material_properties:
                # Add some loss based on material
                material = self.material_properties[profile.material_types[i]]
                ground_loss += 0.1 * material['conductivity']  # Simplified
        
        # Total path loss
        total_loss = fspl_db + diffraction_loss + ground_loss
        
        logger.debug(f"Terrain path loss: FSPL={fspl_db:.2f}dB, Diffraction={diffraction_loss:.2f}dB, Ground={ground_loss:.2f}dB")
        
        return max(0.0, total_loss)
    
    def apply_propagation_effects(self, 
                                 signal: Signal,
                                 rx_platform: Platform,
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply terrain-based propagation effects to a signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            environment: Environmental parameters
            
        Returns:
            Modified signal with propagation effects applied
        """
        # Calculate path loss
        path_loss = self.calculate_path_loss(
            signal.origin, 
            rx_platform.position,
            signal.waveform.center_frequency,
            environment
        )
        
        # Calculate delay (including terrain effects)
        distance = signal.origin.distance_to(rx_platform.position)
        delay = distance / SPEED_OF_LIGHT
        
        # Calculate Doppler shift
        doppler_shift = 0.0
        if distance > 1e-6:
            # Calculate relative velocity vector
            rel_vel_vec = np.array([
                rx_platform.velocity.x - signal.source_velocity.x,
                rx_platform.velocity.y - signal.source_velocity.y,
                rx_platform.velocity.z - signal.source_velocity.z
            ])
            
            # Calculate LOS unit vector
            los_vec = np.array([
                rx_platform.position.x - signal.origin.x,
                rx_platform.position.y - signal.origin.y,
                rx_platform.position.z - signal.origin.z
            ])
            los_unit_vec = los_vec / distance
            
            # Calculate relative speed along LOS
            relative_speed_los = np.dot(rel_vel_vec, los_unit_vec)
            
            # Calculate Doppler shift
            doppler_shift = -(relative_speed_los / SPEED_OF_LIGHT) * signal.waveform.center_frequency
        
        # Create modified signal
        import dataclasses
        modified_signal = dataclasses.replace(
            signal,
            power=signal.power - path_loss,
            propagation_delay=delay,
            doppler_shift=doppler_shift
        )
        
        return modified_signal


class UrbanPropagationModel(IPropagationModel):
    """
    Urban propagation model that accounts for building penetration and urban canyon effects.
    Based on the COST-231 Walfisch-Ikegami model.
    """
    
    def __init__(self):
        """Initialize the urban propagation model."""
        self.buildings: List[BuildingData] = []
        self.building_materials = {
            'concrete': {'penetration_loss': 20.0, 'frequency_factor': 0.3},  # dB
            'glass': {'penetration_loss': 6.0, 'frequency_factor': 0.5},
            'brick': {'penetration_loss': 15.0, 'frequency_factor': 0.4},
            'wood': {'penetration_loss': 4.0, 'frequency_factor': 0.2},
            'metal': {'penetration_loss': 30.0, 'frequency_factor': 0.6},
            'reinforced_concrete': {'penetration_loss': 25.0, 'frequency_factor': 0.45},
            'drywall': {'penetration_loss': 3.0, 'frequency_factor': 0.15},
        }
        
        # Floor/ceiling penetration effects
        self.floor_penetration_loss = 18.0  # dB per floor
        
        # Urban density categories
        self.urban_density_factors = {
            'sparse': 0.7,      # Suburban areas
            'medium': 1.0,      # Standard urban
            'dense': 1.3,       # Dense urban
            'high_rise': 1.5    # High-rise urban centers
        }
    
    def add_building(self, building: BuildingData) -> None:
        """
        Add a building to the urban environment.
        
        Args:
            building: Building data
        """
        self.buildings.append(building)
        logger.debug(f"Added building at {building.position}, height={building.height}m")
    
    def is_line_of_sight(self, tx_position: Position, rx_position: Position) -> bool:
        """
        Check if there is line of sight between transmitter and receiver.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            
        Returns:
            True if there is line of sight, False otherwise
        """
        # Create ray from tx to rx
        direction = np.array([
            rx_position.x - tx_position.x,
            rx_position.y - tx_position.y,
            rx_position.z - tx_position.z
        ])
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
        
        # Check intersection with each building (simplified)
        for building in self.buildings:
            # Simple AABB intersection test
            tx_to_building = np.array([
                building.position.x - tx_position.x,
                building.position.y - tx_position.y,
                building.position.z - tx_position.z
            ])
            
            # Project onto ray direction
            t = np.dot(tx_to_building, direction)
            
            # Skip if building is behind tx or beyond rx
            if t < 0 or t > distance:
                continue
            
            # Find closest point on ray to building center
            closest_point = np.array([
                tx_position.x + t * direction[0],
                tx_position.y + t * direction[1],
                tx_position.z + t * direction[2]
            ])
            
            # Check if point is inside building
            if (abs(closest_point[0] - building.position.x) <= building.width / 2 and
                abs(closest_point[1] - building.position.y) <= building.length / 2 and
                closest_point[2] <= building.position.z + building.height and
                closest_point[2] >= building.position.z):
                return False
        
        return True
    
    def calculate_building_penetration_loss(self, 
                                           tx_position: Position, 
                                           rx_position: Position,
                                           frequency: float,
                                           buildings_in_path: List[BuildingData]) -> float:
        """
        Calculate building penetration loss.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            buildings_in_path: Buildings in the signal path
            
        Returns:
            Building penetration loss in dB
        """
        total_loss = 0.0
        frequency_ghz = frequency / 1e9
        
        for building in buildings_in_path:
            # Get material properties
            if building.material in self.building_materials:
                material = self.building_materials[building.material]
                base_loss = material['penetration_loss']
                
                # Scale loss with frequency (higher frequencies experience more loss)
                freq_scaling = 1.0 + material['frequency_factor'] * (frequency_ghz - 1.0)
                if freq_scaling < 0.5:
                    freq_scaling = 0.5  # Minimum scaling factor
                
                # Calculate penetration distance through building
                # Simplified calculation - assumes straight line through building
                direction = np.array([
                    rx_position.x - tx_position.x,
                    rx_position.y - tx_position.y,
                    rx_position.z - tx_position.z
                ])
                distance = np.linalg.norm(direction)
                direction = direction / distance
                
                # Project onto ray direction to find entry/exit points
                tx_to_building = np.array([
                    building.position.x - tx_position.x,
                    building.position.y - tx_position.y,
                    building.position.z - tx_position.z
                ])
                
                t = np.dot(tx_to_building, direction)
                
                # Calculate penetration distance (simplified)
                penetration_distance = min(building.width, building.length)
                
                # Scale loss by penetration distance
                distance_factor = penetration_distance / 10.0  # Normalize to 10m reference
                
                # Calculate floor penetration if applicable
                floor_loss = 0.0
                if abs(tx_position.z - rx_position.z) > 3.0:  # If significant height difference
                    floors_crossed = max(1, int(abs(tx_position.z - rx_position.z) / 3.0))  # Assume 3m per floor
                    floor_loss = self.floor_penetration_loss * floors_crossed
                
                # Total loss for this building
                building_loss = base_loss * freq_scaling * distance_factor + floor_loss
                total_loss += building_loss
                
                logger.debug(f"Building penetration: material={building.material}, "
                           f"base_loss={base_loss:.1f}dB, freq_factor={freq_scaling:.2f}, "
                           f"distance_factor={distance_factor:.2f}, floor_loss={floor_loss:.1f}dB, "
                           f"total={building_loss:.1f}dB")
        
        return total_loss
    
    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss in urban environment.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB
        """
        distance = tx_position.distance_to(rx_position)
        
        if distance <= 1e-6 or frequency <= 0:
            return float('inf')
        
        # Calculate free space path loss
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * math.pi / SPEED_OF_LIGHT)
        
        # Get urban density from environment parameters or use default
        urban_density = getattr(environment, 'urban_density', 'medium')
        density_factor = self.urban_density_factors.get(urban_density, 1.0)
        
        # Check if there is line of sight
        if self.is_line_of_sight(tx_position, rx_position):
            # LOS path loss (COST-231 Walfisch-Ikegami LOS model)
            distance_km = distance / 1000.0
            frequency_mhz = frequency / 1e6
            
            if distance_km < 0.02:  # Use free space for very short distances
                urban_loss = fspl_db
            else:
                urban_loss = 42.6 + 26 * np.log10(distance_km) + 20 * np.log10(frequency_mhz)
                
            # Apply density factor for LOS (less impact in LOS)
            urban_loss *= (1.0 + (density_factor - 1.0) * 0.3)
        else:
            # NLOS path loss
            distance_km = distance / 1000.0
            frequency_mhz = frequency / 1e6
            
            # Find buildings in path
            buildings_in_path = []
            for building in self.buildings:
                # Simple check if building is between tx and rx
                if not self.is_line_of_sight(tx_position, Position(building.position.x, building.position.y, building.position.z)):
                    buildings_in_path.append(building)
            
            # Calculate building penetration loss
            penetration_loss = self.calculate_building_penetration_loss(
                tx_position, rx_position, frequency, buildings_in_path
            )
            
            # Urban canyon effect (simplified COST-231 Walfisch-Ikegami NLOS model)
            urban_loss = 32.4 + 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz)
            
            # Apply density factor (full impact in NLOS)
            urban_loss *= density_factor
            
            # Add building penetration loss
            urban_loss += penetration_loss
            
            # Add height correction
            h_tx = tx_position.z
            h_rx = rx_position.z
            
            # Height gain/loss
            if h_tx > 50 and h_rx < 10:  # High transmitter, low receiver
                urban_loss -= 5.0  # Slight improvement
            elif h_tx < 10 and h_rx < 10:  # Both low
                urban_loss += 10.0  # Worse propagation
        
        logger.debug(f"Urban path loss: dist={distance:.2f}m, freq={frequency/1e6:.2f}MHz, "
                   f"density={urban_density}, loss={urban_loss:.2f}dB")
        
        return max(0.0, urban_loss)
    
    def apply_propagation_effects(self, 
                                 signal: Signal,
                                 rx_platform: Platform,
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply urban propagation effects to a signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            environment: Environmental parameters
            
        Returns:
            Modified signal with propagation effects applied
        """
        # Calculate path loss
        path_loss = self.calculate_path_loss(
            signal.origin, 
            rx_platform.position,
            signal.waveform.center_frequency,
            environment
        )
        
        # Calculate delay
        distance = signal.origin.distance_to(rx_platform.position)
        
        # Add extra delay for NLOS paths (simplified)
        if not self.is_line_of_sight(signal.origin, rx_platform.position):
            # Add 10% extra distance to account for reflections
            distance *= 1.1
        
        delay = distance / SPEED_OF_LIGHT
        
        # Calculate Doppler shift
        doppler_shift = 0.0
        if distance > 1e-6:
            # Calculate relative velocity vector
            rel_vel_vec = np.array([
                rx_platform.velocity.x - signal.source_velocity.x,
                rx_platform.velocity.y - signal.source_velocity.y,
                rx_platform.velocity.z - signal.source_velocity.z
            ])
            
            # Calculate LOS unit vector
            los_vec = np.array([
                rx_platform.position.x - signal.origin.x,
                rx_platform.position.y - signal.origin.y,
                rx_platform.position.z - signal.origin.z
            ])
            los_unit_vec = los_vec / distance
            
            # Calculate relative speed along LOS
            relative_speed_los = np.dot(rel_vel_vec, los_unit_vec)
            
            # Calculate Doppler shift
            doppler_shift = -(relative_speed_los / SPEED_OF_LIGHT) * signal.waveform.center_frequency
        
        # Create modified signal
        import dataclasses
        modified_signal = dataclasses.replace(
            signal,
            power=signal.power - path_loss,
            propagation_delay=delay,
            doppler_shift=doppler_shift
        )
        
        return modified_signal


class WeatherPropagationModel(IPropagationModel):
    """
    Weather-dependent propagation model that accounts for rain, fog, and atmospheric effects.
    """
    
    def __init__(self, base_model: IPropagationModel):
        """
        Initialize with a base propagation model.
        
        Args:
            base_model: Base propagation model for path loss calculation
        """
        self.base_model = base_model
        
        # Weather condition parameters
        self.weather_conditions = {
            'clear': {
                'rain_rate': 0.0,        # mm/h
                'visibility': 10000.0,   # meters
                'ducting_probability': 0.01,
            },
            'light_rain': {
                'rain_rate': 2.5,        # mm/h
                'visibility': 2000.0,    # meters
                'ducting_probability': 0.05,
            },
            'moderate_rain': {
                'rain_rate': 12.5,       # mm/h
                'visibility': 500.0,     # meters
                'ducting_probability': 0.1,
            },
            'heavy_rain': {
                'rain_rate': 50.0,       # mm/h
                'visibility': 100.0,     # meters
                'ducting_probability': 0.2,
            },
            'fog': {
                'rain_rate': 0.0,        # mm/h
                'visibility': 50.0,      # meters
                'ducting_probability': 0.3,
            },
            'sea_fog': {
                'rain_rate': 0.0,        # mm/h
                'visibility': 30.0,      # meters
                'ducting_probability': 0.6,
            }
        }
    
    def calculate_rain_attenuation(self, distance: float, frequency: float, rain_rate: float) -> float:
        """
        Calculate rain attenuation.
        
        Args:
            distance: Path length in meters
            frequency: Signal frequency in Hz
            rain_rate: Rain rate in mm/h
            
        Returns:
            Rain attenuation in dB
        """
        # ITU-R P.838 simplified model
        frequency_ghz = frequency / 1e9
        
        # k and α coefficients (simplified)
        if frequency_ghz < 10:
            k = 0.0001 * frequency_ghz
            alpha = 1.0
        elif frequency_ghz < 40:
            k = 0.01 * frequency_ghz - 0.1
            alpha = 1.2
        else:
            k = 0.3
            alpha = 1.0
        
        # Specific attenuation (dB/km)
        specific_attenuation = k * (rain_rate ** alpha)
        
        # Total attenuation
        distance_km = distance / 1000.0
        attenuation = specific_attenuation * distance_km
        
        return attenuation
    
    def calculate_fog_attenuation(self, distance: float, frequency: float, visibility: float) -> float:
        """
        Calculate fog/cloud attenuation.
        
        Args:
            distance: Path length in meters
            frequency: Signal frequency in Hz
            visibility: Visibility in meters
            
        Returns:
            Fog attenuation in dB
        """
        # Simplified model based on visibility
        if visibility > 1000:  # Clear conditions
            return 0.0
        
        frequency_ghz = frequency / 1e9
        
        # Specific attenuation (dB/km) - simplified model
        if visibility < 50:  # Dense fog
            specific_attenuation = 0.4 * frequency_ghz
        elif visibility < 200:  # Moderate fog
            specific_attenuation = 0.2 * frequency_ghz
        else:  # Light fog
            specific_attenuation = 0.1 * frequency_ghz
        
        # Total attenuation
        distance_km = distance / 1000.0
        attenuation = specific_attenuation * distance_km
        
        return attenuation
    
    def calculate_atmospheric_attenuation(self, distance: float, frequency: float, 
                                         temperature: float, humidity: float, 
                                         pressure: float) -> float:
        """
        Calculate atmospheric attenuation due to oxygen and water vapor.
        
        Args:
            distance: Path length in meters
            frequency: Signal frequency in Hz
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            pressure: Atmospheric pressure in hPa
            
        Returns:
            Atmospheric attenuation in dB
        """
        # Simplified model based on ITU-R P.676
        frequency_ghz = frequency / 1e9
        
        # Oxygen attenuation (dB/km) - simplified
        if frequency_ghz < 50:
            oxygen_attenuation = 0.01 * (frequency_ghz / 10)
        elif frequency_ghz < 70:
            oxygen_attenuation = 0.1 + (frequency_ghz - 50) * 0.01
        else:
            oxygen_attenuation = 0.3
        
        # Water vapor attenuation (dB/km) - simplified
        water_vapor_density = humidity * 0.1 * np.exp(0.06 * temperature)  # g/m³, simplified
        
        if frequency_ghz < 10:
            water_vapor_attenuation = 0.0001 * frequency_ghz * water_vapor_density
        elif frequency_ghz < 100:
            water_vapor_attenuation = 0.001 * frequency_ghz * water_vapor_density
        else:
            water_vapor_attenuation = 0.1 * water_vapor_density
        
        # Total atmospheric attenuation
        distance_km = distance / 1000.0
        attenuation = (oxygen_attenuation + water_vapor_attenuation) * distance_km
        
        return attenuation
    
    def calculate_ducting_effect(self, 
                               tx_position: Position, 
                               rx_position: Position,
                               frequency: float,
                               ducting_probability: float) -> float:
        """
        Calculate atmospheric ducting effect.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            ducting_probability: Probability of ducting conditions
            
        Returns:
            Path loss modification in dB (negative values = enhancement)
        """
        # Only apply ducting effects for certain conditions
        if ducting_probability < 0.01:
            return 0.0
            
        distance = tx_position.distance_to(rx_position)
        frequency_ghz = frequency / 1e9
        
        # Check if both tx and rx are in potential ducting layer
        # Simplified: assume ducting occurs below 100m altitude
        if tx_position.z < 100 and rx_position.z < 100:
            # Ducting typically enhances propagation (reduces path loss)
            # Effect increases with distance and is more pronounced at higher frequencies
            
            # Random factor based on probability
            if np.random.random() < ducting_probability:
                # Calculate enhancement (negative dB value)
                # More enhancement for longer distances
                distance_km = distance / 1000.0
                
                # Simplified model: enhancement increases with distance up to a point
                if distance_km < 5:
                    enhancement = -2.0 * distance_km * ducting_probability
                elif distance_km < 50:
                    enhancement = -10.0 * ducting_probability * (1 + 0.1 * frequency_ghz)
                else:
                    # Beyond certain distance, enhancement diminishes
                    enhancement = -10.0 * ducting_probability * (1 + 0.1 * frequency_ghz) * (1 - (distance_km - 50) / 100)
                    
                logger.debug(f"Ducting enhancement: {enhancement:.2f} dB at {distance_km:.1f} km")
                return enhancement
        
        return 0.0
    
    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss with weather effects.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB
        """
        # Get base path loss from underlying model
        base_path_loss = self.base_model.calculate_path_loss(
            tx_position, rx_position, frequency, environment
        )
        
        # Get weather condition from environment or use default
        weather_condition = getattr(environment, 'weather_condition', 'clear')
        
        # Get specific parameters for this weather condition
        if weather_condition in self.weather_conditions:
            params = self.weather_conditions[weather_condition]
        else:
            # Default to clear weather
            params = self.weather_conditions['clear']
            
        # Get distance
        distance = tx_position.distance_to(rx_position)
        distance_km = distance / 1000.0
        
        # Calculate additional attenuation due to rain
        rain_attenuation = self.calculate_rain_attenuation(
            distance, frequency, params['rain_rate']
        )
        
        # Calculate additional attenuation due to fog
        fog_attenuation = self.calculate_fog_attenuation(
            distance, frequency, params['visibility']
        )
        
        # Calculate atmospheric attenuation
        temperature = getattr(environment, 'temperature', 290.0) - 273.15  # Convert K to C
        humidity = getattr(environment, 'humidity', 50.0)
        pressure = getattr(environment, 'pressure', 1013.25)
        
        atmospheric_attenuation = self.calculate_atmospheric_attenuation(
            distance, frequency, temperature, humidity, pressure
        )
        
        # Calculate ducting effect (can be negative = enhancement)
        ducting_effect = self.calculate_ducting_effect(
            tx_position, rx_position, frequency, params['ducting_probability']
        )
        
        # Total path loss
        total_path_loss = base_path_loss + rain_attenuation + fog_attenuation + atmospheric_attenuation + ducting_effect
        
        logger.debug(f"Weather effects: rain={rain_attenuation:.2f}dB, fog={fog_attenuation:.2f}dB, "
                   f"atm={atmospheric_attenuation:.2f}dB, ducting={ducting_effect:.2f}dB, "
                   f"weather={weather_condition}, total_add={rain_attenuation+fog_attenuation+atmospheric_attenuation+ducting_effect:.2f}dB")
        
        return max(0.0, total_path_loss)
    
    def apply_propagation_effects(self, 
                                 signal: Signal,
                                 rx_platform: Platform,
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply weather-dependent propagation effects to a signal.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            environment: Environmental parameters
            
        Returns:
            Modified signal with weather effects applied
        """
        # Calculate path loss with weather effects
        path_loss = self.calculate_path_loss(
            signal.origin, rx_platform.position, signal.waveform.center_frequency, environment
        )
        
        # Get base signal with other effects from base model
        base_signal = self.base_model.apply_propagation_effects(signal, rx_platform, environment)
        
        # Apply our path loss instead of the base model's
        import dataclasses
        modified_signal = dataclasses.replace(
            base_signal,
            power=signal.power - path_loss
        )
        
        return modified_signal