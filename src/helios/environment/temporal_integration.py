"""Integration module for temporal variations in RF environment."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import copy

from helios.core.data_structures import Position, Orientation
from helios.environment.rcs import RCSModel
from helios.environment.clutter import ClutterPoint, ClutterPatch
from helios.environment.continuous_clutter import TerrainPatch
from helios.environment.temporal_rcs import TemporalRCSModel
from helios.environment.temporal_clutter import TemporalClutterModel
from helios.utils.logger import get_logger

logger = get_logger(__name__)

# Common temporal variation functions

def sinusoidal_rcs_variation(time: float, base_rcs: float, amplitude: float = 0.2, period: float = 1.0) -> float:
    """
    Sinusoidal RCS variation.
    
    Args:
        time: Current time (seconds)
        base_rcs: Base RCS value
        amplitude: Variation amplitude (fraction of base RCS)
        period: Variation period (seconds)
        
    Returns:
        Modified RCS value
    """
    return base_rcs * (1.0 + amplitude * np.sin(2 * np.pi * time / period))

def random_walk_rcs_variation(time: float, base_rcs: float, step_size: float = 0.05, seed: Optional[int] = None) -> float:
    """
    Random walk RCS variation.
    
    Args:
        time: Current time (seconds)
        base_rcs: Base RCS value
        step_size: Maximum step size (fraction of base RCS)
        seed: Random seed
        
    Returns:
        Modified RCS value
    """
    # Use deterministic random based on time and seed
    rng = np.random.RandomState(seed=int(time * 1000) if seed is None else seed)
    
    # Generate random walk step
    step = rng.uniform(-step_size, step_size) * base_rcs
    
    return base_rcs + step

def weather_affected_clutter(time: float, base_point: ClutterPoint, 
                           rain_intensity: float = 0.0, 
                           wind_speed: float = 0.0) -> ClutterPoint:
    """
    Weather-affected clutter point variation.
    
    Args:
        time: Current time (seconds)
        base_point: Base clutter point
        rain_intensity: Rain intensity (0-1)
        wind_speed: Wind speed (m/s)
        
    Returns:
        Modified clutter point
    """
    # Create a copy to avoid modifying the original
    point = copy.deepcopy(base_point)
    
    # Increase RCS with rain (water has higher reflectivity)
    rain_factor = 1.0 + rain_intensity * 0.5
    
    # Add small position variations due to wind
    wind_displacement = wind_speed * 0.01 * np.sin(time)
    
    # Apply modifications
    point.rcs *= rain_factor
    point.position.x += wind_displacement
    
    return point

def create_temporal_rcs_model(base_model: RCSModel, 
                             variation_type: str = "sinusoidal",
                             period: float = 1.0,
                             amplitude: float = 0.2) -> RCSModel:
    """
    Create a temporal RCS model from a base model.
    
    Args:
        base_model: Base RCS model
        variation_type: Type of variation ("sinusoidal" or "random_walk")
        period: Variation period (seconds)
        amplitude: Variation amplitude (fraction of base RCS)
        
    Returns:
        Temporal RCS model
    """
    if variation_type == "sinusoidal":
        def variation_func(t: float) -> float:
            return 1.0 + amplitude * np.sin(2 * np.pi * t / period)
    elif variation_type == "random_walk":
        def variation_func(t: float) -> float:
            rng = np.random.RandomState(seed=int(t * 1000))
            return 1.0 + rng.uniform(-amplitude, amplitude)
    else:
        logger.warning(f"Unknown variation type: {variation_type}. Using sinusoidal.")
        def variation_func(t: float) -> float:
            return 1.0 + amplitude * np.sin(2 * np.pi * t / period)
    
    # Create a wrapper that adds temporal behavior
    class TemporalRCSWrapper(RCSModel):
        def __init__(self, base_model: RCSModel):
            super().__init__(base_model.model_type, base_model.base_rcs)
            self.base_model = base_model
            
        def calculate_rcs(self, frequency: float, target_orientation: Orientation,
                         incident_direction: Tuple[float, float, float], 
                         current_time: Optional[float] = None) -> float:
            base_rcs = self.base_model.calculate_rcs(frequency, target_orientation, incident_direction)
            if current_time is None:
                current_time = 0.0
            return base_rcs * variation_func(current_time)
    
    return TemporalRCSWrapper(base_model)