"""Time-varying RCS models for dynamic platforms."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import time

from helios.core.data_structures import Position, Orientation
from helios.environment.rcs import RCSModel, RCSModelType
from helios.environment.rcs_integration import ImportedRCSModel
from helios.environment.rcs_multistatic import RCSDataFormat
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class TemporalRCSModel(ImportedRCSModel):
    """
    RCS model with temporal variations to simulate dynamic platform characteristics.
    Extends ImportedRCSModel to add time-dependent behavior.
    """
    
    def __init__(self, 
                data_file: str, 
                format_type: RCSDataFormat = RCSDataFormat.CSV,
                name: str = "temporal_rcs",
                variation_function: Optional[Callable[[float], float]] = None,
                variation_period: float = 1.0,
                variation_amplitude: float = 0.2):
        """
        Initialize the temporal RCS model.
        
        Args:
            data_file: Path to the RCS data file
            format_type: Format of the data file
            name: Name identifier for this RCS model
            variation_function: Custom function for time variation (takes time in seconds, returns multiplier)
            variation_period: Period of variation in seconds
            variation_amplitude: Amplitude of variation (fraction of base RCS)
        """
        super().__init__(data_file, format_type, name)
        self.variation_period = variation_period
        self.variation_amplitude = variation_amplitude
        self.variation_function = variation_function or self._default_variation
        self.start_time = time.time()
    
    def _default_variation(self, t: float) -> float:
        """
        Default sinusoidal variation function.
        
        Args:
            t: Time in seconds
            
        Returns:
            Multiplier for RCS (centered around 1.0)
        """
        # Sinusoidal variation with period and amplitude
        return 1.0 + self.variation_amplitude * np.sin(2 * np.pi * t / self.variation_period)
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float],
                     current_time: Optional[float] = None) -> float:
        """
        Calculate time-varying RCS.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            current_time: Current simulation time (seconds)
            
        Returns:
            RCS value in m²
        """
        # Get base RCS from parent class
        base_rcs = super().calculate_rcs(frequency, target_orientation, incident_direction)
        
        # Apply temporal variation
        if current_time is None:
            current_time = time.time() - self.start_time
            
        variation_factor = self.variation_function(current_time)
        
        return base_rcs * variation_factor
    
    def calculate_bistatic_rcs(self,
                              frequency: float,
                              target_orientation: Orientation,
                              incident_direction: Tuple[float, float, float],
                              observation_direction: Tuple[float, float, float],
                              current_time: Optional[float] = None) -> float:
        """
        Calculate time-varying bistatic RCS.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            observation_direction: Direction to observer (unit vector)
            current_time: Current simulation time (seconds)
            
        Returns:
            RCS value in m²
        """
        # Get base bistatic RCS
        base_rcs = super().calculate_bistatic_rcs(
            frequency, target_orientation, incident_direction, observation_direction
        )
        
        # Apply temporal variation
        if current_time is None:
            current_time = time.time() - self.start_time
            
        variation_factor = self.variation_function(current_time)
        
        return base_rcs * variation_factor