"""Multi-static Radar Cross Section (RCS) models with external data import capabilities."""

import numpy as np
import os
import json
import csv
from enum import Enum
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
from dataclasses import dataclass
import scipy.interpolate as interp

from helios.core.data_structures import Position, Orientation
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class RCSDataFormat(Enum):
    """Supported formats for RCS data import."""
    CSV = "csv"
    JSON = "json"
    NPY = "npy"
    CUSTOM = "custom"


@dataclass
class RCSDataPoint:
    """Data point for RCS measurements or simulations."""
    # Incident angles (degrees)
    azimuth: float
    elevation: float
    # Observation angles (degrees)
    obs_azimuth: float
    obs_elevation: float
    # Frequency (Hz)
    frequency: float
    # RCS value (m²)
    rcs: float
    # Optional polarization information
    polarization: str = "HH"  # HH, VV, HV, VH


class MultistaticRCSModel:
    """
    RCS model that supports multi-static scenarios with data imported from external sources.
    
    This model handles:
    - Bistatic and multistatic RCS calculations
    - Frequency-dependent RCS
    - Angle-dependent RCS (both incident and observation angles)
    - Interpolation between measured/simulated data points
    - Multiple data formats
    """
    
    def __init__(self, name: str = "generic_target"):
        """
        Initialize the multistatic RCS model.
        
        Args:
            name: Name identifier for this RCS model
        """
        self.name = name
        self.data_points: List[RCSDataPoint] = []
        self.interpolator = None
        self.frequency_range: Tuple[float, float] = (0, 0)
        self.azimuth_range: Tuple[float, float] = (0, 0)
        self.elevation_range: Tuple[float, float] = (0, 0)
        self.is_initialized = False
        
    def import_data(self, 
                   file_path: str, 
                   format_type: RCSDataFormat = RCSDataFormat.CSV,
                   custom_parser: Optional[Callable] = None) -> bool:
        """
        Import RCS data from external file.
        
        Args:
            file_path: Path to the data file
            format_type: Format of the data file
            custom_parser: Custom parser function for specialized formats
            
        Returns:
            Success status
        """
        if not os.path.exists(file_path):
            logger.error(f"RCS data file not found: {file_path}")
            return False
            
        try:
            if format_type == RCSDataFormat.CSV:
                self._import_csv(file_path)
            elif format_type == RCSDataFormat.JSON:
                self._import_json(file_path)
            elif format_type == RCSDataFormat.NPY:
                self._import_numpy(file_path)
            elif format_type == RCSDataFormat.CUSTOM and custom_parser:
                self.data_points = custom_parser(file_path)
            else:
                logger.error(f"Unsupported format or missing custom parser: {format_type}")
                return False
                
            self._initialize_interpolator()
            return True
            
        except Exception as e:
            logger.error(f"Error importing RCS data: {str(e)}")
            return False
    
    def _import_csv(self, file_path: str) -> None:
        """Import RCS data from CSV file."""
        self.data_points = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    point = RCSDataPoint(
                        azimuth=float(row.get('azimuth', 0)),
                        elevation=float(row.get('elevation', 0)),
                        obs_azimuth=float(row.get('obs_azimuth', row.get('azimuth', 0))),
                        obs_elevation=float(row.get('obs_elevation', row.get('elevation', 0))),
                        frequency=float(row.get('frequency', 0)),
                        rcs=float(row.get('rcs', 0)),
                        polarization=row.get('polarization', 'HH')
                    )
                    self.data_points.append(point)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row in CSV: {row}, error: {str(e)}")
    
    def _import_json(self, file_path: str) -> None:
        """Import RCS data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        self.data_points = []
        for item in data:
            try:
                point = RCSDataPoint(
                    azimuth=float(item.get('azimuth', 0)),
                    elevation=float(item.get('elevation', 0)),
                    obs_azimuth=float(item.get('obs_azimuth', item.get('azimuth', 0))),
                    obs_elevation=float(item.get('obs_elevation', item.get('elevation', 0))),
                    frequency=float(item.get('frequency', 0)),
                    rcs=float(item.get('rcs', 0)),
                    polarization=item.get('polarization', 'HH')
                )
                self.data_points.append(point)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid item in JSON: {item}, error: {str(e)}")
    
    def _import_numpy(self, file_path: str) -> None:
        """Import RCS data from NumPy file."""
        data = np.load(file_path)
        self.data_points = []
        
        # Expect structured array with fields: azimuth, elevation, obs_azimuth, obs_elevation, frequency, rcs
        if isinstance(data, np.ndarray):
            for row in data:
                try:
                    # Handle structured arrays
                    if data.dtype.names:
                        point = RCSDataPoint(
                            azimuth=float(row['azimuth']),
                            elevation=float(row['elevation']),
                            obs_azimuth=float(row.get('obs_azimuth', row['azimuth'])),
                            obs_elevation=float(row.get('obs_elevation', row['elevation'])),
                            frequency=float(row['frequency']),
                            rcs=float(row['rcs']),
                            polarization=str(row.get('polarization', 'HH'))
                        )
                    # Handle regular arrays with fixed column order
                    else:
                        point = RCSDataPoint(
                            azimuth=float(row[0]),
                            elevation=float(row[1]),
                            obs_azimuth=float(row[2] if len(row) > 2 else row[0]),
                            obs_elevation=float(row[3] if len(row) > 3 else row[1]),
                            frequency=float(row[4] if len(row) > 4 else 1e9),
                            rcs=float(row[5] if len(row) > 5 else row[2]),
                            polarization='HH'
                        )
                    self.data_points.append(point)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid row in NumPy data: {row}, error: {str(e)}")
    
    def _initialize_interpolator(self) -> None:
        """Initialize the interpolator for RCS data."""
        if not self.data_points:
            logger.warning("No data points available for interpolation")
            self.is_initialized = False
            return
            
        # Extract data for interpolation
        points = []
        values = []
        
        # Find ranges for all parameters
        azimuths = []
        elevations = []
        obs_azimuths = []
        obs_elevations = []
        frequencies = []
        
        for point in self.data_points:
            azimuths.append(point.azimuth)
            elevations.append(point.elevation)
            obs_azimuths.append(point.obs_azimuth)
            obs_elevations.append(point.obs_elevation)
            frequencies.append(point.frequency)
            
            # Create 5D point: [azimuth, elevation, obs_azimuth, obs_elevation, frequency]
            points.append([
                point.azimuth, 
                point.elevation, 
                point.obs_azimuth, 
                point.obs_elevation, 
                point.frequency
            ])
            values.append(point.rcs)
        
        # Set ranges
        self.azimuth_range = (min(azimuths), max(azimuths))
        self.elevation_range = (min(elevations), max(elevations))
        self.frequency_range = (min(frequencies), max(frequencies))
        
        # Convert to numpy arrays
        points_array = np.array(points)
        values_array = np.array(values)
        
        # Create interpolator (using linear interpolation)
        try:
            self.interpolator = interp.LinearNDInterpolator(points_array, values_array)
            # Fallback for points outside the convex hull
            self.nearest_interpolator = interp.NearestNDInterpolator(points_array, values_array)
            self.is_initialized = True
            logger.info(f"RCS interpolator initialized with {len(self.data_points)} data points")
        except Exception as e:
            logger.error(f"Failed to initialize interpolator: {str(e)}")
            self.is_initialized = False
    
    def calculate_rcs(self, 
                     frequency: float,
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float],
                     observation_direction: Optional[Tuple[float, float, float]] = None) -> float:
        """
        Calculate RCS for given parameters using interpolation of imported data.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            observation_direction: Direction to observer (unit vector), defaults to backscatter
            
        Returns:
            RCS value in m²
        """
        if not self.is_initialized or not self.interpolator:
            logger.warning("RCS model not initialized with data")
            return 1.0  # Default RCS
            
        # Default to backscatter if observation direction not provided
        if observation_direction is None:
            # Invert incident direction for backscatter
            observation_direction = (-incident_direction[0], 
                                    -incident_direction[1], 
                                    -incident_direction[2])
        
        # Convert directions to angles
        incident_azimuth, incident_elevation = self._vector_to_angles(incident_direction)
        obs_azimuth, obs_elevation = self._vector_to_angles(observation_direction)
        
        # Apply target orientation to adjust angles
        incident_azimuth, incident_elevation = self._adjust_angles_for_orientation(
            incident_azimuth, incident_elevation, target_orientation
        )
        obs_azimuth, obs_elevation = self._adjust_angles_for_orientation(
            obs_azimuth, obs_elevation, target_orientation
        )
        
        # Ensure frequency is within range
        freq = np.clip(frequency, self.frequency_range[0], self.frequency_range[1])
        
        # Create query point
        query = np.array([
            incident_azimuth, 
            incident_elevation, 
            obs_azimuth, 
            obs_elevation, 
            freq
        ])
        
        # Interpolate RCS value
        rcs = self.interpolator(query)
        
        # Use nearest neighbor if outside convex hull
        if np.isnan(rcs):
            rcs = self.nearest_interpolator(query)
            
        # Ensure positive RCS
        return max(float(rcs), 0.01)
    
    def _vector_to_angles(self, direction: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Convert a direction vector to azimuth and elevation angles.
        
        Args:
            direction: 3D direction vector
            
        Returns:
            Tuple of (azimuth, elevation) in degrees
        """
        x, y, z = direction
        
        # Calculate azimuth (0° is along x-axis, positive clockwise in x-y plane)
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Ensure azimuth is in [0, 360)
        if azimuth < 0:
            azimuth += 360
            
        # Calculate elevation (0° is in x-y plane, positive up)
        r_xy = np.sqrt(x**2 + y**2)
        elevation = np.degrees(np.arctan2(z, r_xy))
        
        return azimuth, elevation
    
    def _adjust_angles_for_orientation(self, 
                                      azimuth: float, 
                                      elevation: float, 
                                      orientation: Orientation) -> Tuple[float, float]:
        """
        Adjust incident/observation angles based on target orientation.
        
        Args:
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            orientation: Target orientation
            
        Returns:
            Adjusted (azimuth, elevation) in degrees
        """
        # Convert angles to radians
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Convert spherical to cartesian
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        # Create rotation matrices for roll, pitch, yaw
        # This is a simplified rotation calculation
        cos_roll = np.cos(orientation.roll)
        sin_roll = np.sin(orientation.roll)
        cos_pitch = np.cos(orientation.pitch)
        sin_pitch = np.sin(orientation.pitch)
        cos_yaw = np.cos(orientation.yaw)
        sin_yaw = np.sin(orientation.yaw)
        
        # Apply rotations (simplified)
        # First yaw
        x_temp = x * cos_yaw - y * sin_yaw
        y_temp = x * sin_yaw + y * cos_yaw
        x, y = x_temp, y_temp
        
        # Then pitch
        x_temp = x * cos_pitch + z * sin_pitch
        z_temp = -x * sin_pitch + z * cos_pitch
        x, z = x_temp, z_temp
        
        # Then roll
        y_temp = y * cos_roll - z * sin_roll
        z_temp = y * sin_roll + z * cos_roll
        y, z = y_temp, z_temp
        
        # Convert back to spherical
        adjusted_azimuth = np.degrees(np.arctan2(y, x))
        if adjusted_azimuth < 0:
            adjusted_azimuth += 360
            
        r_xy = np.sqrt(x**2 + y**2)
        adjusted_elevation = np.degrees(np.arctan2(z, r_xy))
        
        return adjusted_azimuth, adjusted_elevation