"""Time-varying clutter models for dynamic environments."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import time

from helios.core.data_structures import Position, Signal, Platform, EnvironmentParameters
from helios.environment.clutter import ClutterPoint, ClutterPatch, DiscreteClutterModel
from helios.environment.continuous_clutter import TerrainPatch, ContinuousClutterModel
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class TemporalClutterModel(ContinuousClutterModel):
    """
    Clutter model with temporal variations to simulate dynamic environments.
    Extends ContinuousClutterModel to add time-dependent behavior.
    """
    
    def __init__(self):
        """Initialize the temporal clutter model."""
        super().__init__()
        self.time_varying_points: Dict[str, Dict[str, Any]] = {}
        self.time_varying_patches: Dict[str, Dict[str, Any]] = {}
        self.time_varying_terrain: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
    
    def add_time_varying_point(self, 
                              point: ClutterPoint, 
                              variation_function: Callable[[float, ClutterPoint], ClutterPoint],
                              period: float = 1.0) -> None:
        """
        Add a time-varying clutter point.
        
        Args:
            point: Base clutter point
            variation_function: Function that takes (time, point) and returns modified point
            period: Period of variation in seconds
        """
        self.clutter_points.append(point)
        self.time_varying_points[point.id] = {
            'function': variation_function,
            'period': period,
            'base_point': point
        }
        logger.debug(f"Added time-varying clutter point at {point.position}, RCS={point.rcs} m²")
    
    def add_time_varying_patch(self, 
                              patch: ClutterPatch, 
                              variation_function: Callable[[float, ClutterPatch], ClutterPatch],
                              period: float = 1.0) -> None:
        """
        Add a time-varying clutter patch.
        
        Args:
            patch: Base clutter patch
            variation_function: Function that takes (time, patch) and returns modified patch
            period: Period of variation in seconds
        """
        self.clutter_patches.append(patch)
        self.time_varying_patches[patch.id] = {
            'function': variation_function,
            'period': period,
            'base_patch': patch
        }
        logger.debug(f"Added time-varying clutter patch at {patch.center}")
    
    def add_time_varying_terrain(self, 
                                patch: TerrainPatch, 
                                variation_function: Callable[[float, TerrainPatch], TerrainPatch],
                                period: float = 1.0) -> None:
        """
        Add a time-varying terrain patch.
        
        Args:
            patch: Base terrain patch
            variation_function: Function that takes (time, patch) and returns modified patch
            period: Period of variation in seconds
        """
        self.terrain_patches.append(patch)
        self.time_varying_terrain[patch.id] = {
            'function': variation_function,
            'period': period,
            'base_patch': patch
        }
        logger.debug(f"Added time-varying terrain patch at {patch.center}")
    
    def update(self, current_time: Optional[float] = None) -> None:
        """
        Update all time-varying clutter elements.
        
        Args:
            current_time: Current simulation time (seconds)
        """
        if current_time is None:
            current_time = time.time() - self.start_time
        
        # Update time-varying points
        for point_id, data in self.time_varying_points.items():
            # Find the point in the list
            for i, point in enumerate(self.clutter_points):
                if point.id == point_id:
                    # Apply variation function
                    self.clutter_points[i] = data['function'](current_time, data['base_point'])
                    break
        
        # Update time-varying patches
        for patch_id, data in self.time_varying_patches.items():
            for i, patch in enumerate(self.clutter_patches):
                if patch.id == patch_id:
                    self.clutter_patches[i] = data['function'](current_time, data['base_patch'])
                    break
        
        # Update time-varying terrain
        for patch_id, data in self.time_varying_terrain.items():
            for i, patch in enumerate(self.terrain_patches):
                if patch.id == patch_id:
                    self.terrain_patches[i] = data['function'](current_time, data['base_patch'])
                    break
    
    def generate_reflections(self, signal: Signal, rx_platform: Platform, current_time: Optional[float] = None) -> List[Signal]:
        """
        Generate reflected signals from time-varying clutter.
        
        Args:
            signal: Original transmitted signal
            rx_platform: Receiving platform
            current_time: Current simulation time (seconds)
            
        Returns:
            List of reflected signals
        """
        # Update clutter elements based on current time
        self.update(current_time)
        
        # Use parent method to generate reflections
        return super().generate_reflections(signal, rx_platform)