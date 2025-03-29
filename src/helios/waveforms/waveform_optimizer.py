"""Waveform optimization algorithms for specific tactical scenarios."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, TypeVar, cast
from enum import Enum, auto
import copy

from helios.core.data_structures import Waveform
from helios.waveforms.tactical_waveforms import (
    Link16Waveform, SINCGARSWaveform, HaveQuickWaveform, MILSTDWaveform
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

# Define type variables for each waveform type
T = TypeVar('T', bound=Waveform)

class OptimizationGoal(Enum):
    """Optimization goals for tactical waveforms."""
    LOW_PROBABILITY_OF_DETECTION = auto()
    ANTI_JAMMING = auto()
    MAXIMUM_RANGE = auto()
    MINIMUM_BIT_ERROR_RATE = auto()
    COVERT_OPERATION = auto()

class WaveformOptimizer:
    """Optimizes waveforms for specific tactical scenarios."""
    
    def optimize(self, waveform: T, goal: OptimizationGoal, 
                 constraints: Optional[Dict[str, Any]] = None) -> T:
        """
        Optimize a waveform for a specific goal.
        
        Args:
            waveform: The waveform to optimize
            goal: The optimization goal
            constraints: Optional constraints on optimization
            
        Returns:
            Optimized waveform
        """
        logger.info(f"Optimizing waveform for {goal.name}")
        
        # Create a deep copy of the waveform to modify
        optimized = copy.deepcopy(waveform)
        
        # Apply optimization based on waveform type and goal
        if isinstance(optimized, Link16Waveform):
            return self._optimize_link16(optimized, goal, constraints)
        elif isinstance(optimized, SINCGARSWaveform):
            return self._optimize_sincgars(optimized, goal, constraints)
        elif isinstance(optimized, HaveQuickWaveform):
            return self._optimize_have_quick(optimized, goal, constraints)
        elif isinstance(optimized, MILSTDWaveform):
            return self._optimize_milstd(optimized, goal, constraints)
        else:
            # Generic optimization for other waveform types
            return self._optimize_generic(optimized, goal, constraints)
    
    def _optimize_link16(self, waveform: Link16Waveform, goal: OptimizationGoal,
                        constraints: Optional[Dict[str, Any]]) -> Link16Waveform:
        """Optimize Link-16 waveform."""
        if goal == OptimizationGoal.LOW_PROBABILITY_OF_DETECTION:
            # Increase frequency hopping security level
            waveform.transmission_security_level = 3
            waveform.frequency_hopping_pattern = "secure"
            # Reduce power when possible
            if not constraints or 'min_amplitude' not in constraints:
                waveform.amplitude *= 0.7
        
        elif goal == OptimizationGoal.ANTI_JAMMING:
            # Use more secure hopping pattern
            waveform.frequency_hopping_pattern = "secure"
            # Increase PRF for better jamming resistance
            waveform.pulse_repetition_frequency = 75e3
        
        elif goal == OptimizationGoal.MAXIMUM_RANGE:
            # Increase power
            waveform.amplitude *= 1.5
            # Use more robust message format
            waveform.message_format = "MIDS"
            
        return waveform
    
    def _optimize_sincgars(self, waveform: SINCGARSWaveform, goal: OptimizationGoal,
                          constraints: Optional[Dict[str, Any]]) -> SINCGARSWaveform:
        """Optimize SINCGARS waveform."""
        if goal == OptimizationGoal.LOW_PROBABILITY_OF_DETECTION:
            # Increase hop rate for better LPD
            waveform.hop_rate = 200.0
            # Reduce power when possible
            if not constraints or 'min_amplitude' not in constraints:
                waveform.amplitude *= 0.7
        
        elif goal == OptimizationGoal.ANTI_JAMMING:
            # Increase hop rate for better anti-jamming
            waveform.hop_rate = 300.0
            # Ensure frequency hopping mode
            waveform.mode = "frequency_hopping"
            
        elif goal == OptimizationGoal.MAXIMUM_RANGE:
            # Increase power
            waveform.amplitude *= 1.5
            # Use single channel mode for better SNR
            waveform.mode = "single_channel"
            
        return waveform
    
    def _optimize_have_quick(self, waveform: HaveQuickWaveform, goal: OptimizationGoal,
                            constraints: Optional[Dict[str, Any]]) -> HaveQuickWaveform:
        """Optimize HAVE QUICK waveform."""
        if goal == OptimizationGoal.LOW_PROBABILITY_OF_DETECTION:
            # Decrease dwell time
            waveform.dwell_time = 0.005
            # Reduce power when possible
            if not constraints or 'min_amplitude' not in constraints:
                waveform.amplitude *= 0.7
        
        elif goal == OptimizationGoal.ANTI_JAMMING:
            # Decrease dwell time for faster frequency changes
            waveform.dwell_time = 0.003
            # Increase channel set for more frequency options
            waveform.channel_set = 3
            
        elif goal == OptimizationGoal.MAXIMUM_RANGE:
            # Increase power
            waveform.amplitude *= 1.5
            # Increase dwell time for better SNR
            waveform.dwell_time = 0.02
            
        return waveform
    
    def _optimize_milstd(self, waveform: MILSTDWaveform, goal: OptimizationGoal,
                        constraints: Optional[Dict[str, Any]]) -> MILSTDWaveform:
        """Optimize MIL-STD waveform."""
        if goal == OptimizationGoal.LOW_PROBABILITY_OF_DETECTION:
            # Use robust mode
            waveform.robust_mode = True
            # Reduce data rate for better error correction
            waveform.data_rate = 600.0
            # Reduce power when possible
            if not constraints or 'min_amplitude' not in constraints:
                waveform.amplitude *= 0.7
        
        elif goal == OptimizationGoal.ANTI_JAMMING:
            # Enable FEC
            waveform.forward_error_correction = True
            # Use long interleaver
            waveform.interleaver_type = "long"
            # Use more robust standard
            waveform.standard_type = "188-110B"
            
        elif goal == OptimizationGoal.MINIMUM_BIT_ERROR_RATE:
            # Enable FEC
            waveform.forward_error_correction = True
            # Use long interleaver
            waveform.interleaver_type = "long"
            # Reduce data rate for better error correction
            waveform.data_rate = 600.0
            
        return waveform
    
    def _optimize_generic(self, waveform: Waveform, goal: OptimizationGoal,
                         constraints: Optional[Dict[str, Any]]) -> Waveform:
        """Generic optimization for any waveform type."""
        if goal == OptimizationGoal.LOW_PROBABILITY_OF_DETECTION:
            # Reduce power when possible
            if not constraints or 'min_amplitude' not in constraints:
                waveform.amplitude *= 0.7
            # Increase bandwidth if possible
            if not constraints or 'max_bandwidth' not in constraints:
                waveform.bandwidth *= 1.5
                
        elif goal == OptimizationGoal.ANTI_JAMMING:
            # Increase bandwidth if possible
            if not constraints or 'max_bandwidth' not in constraints:
                waveform.bandwidth *= 2.0
                
        elif goal == OptimizationGoal.MAXIMUM_RANGE:
            # Increase power
            waveform.amplitude *= 1.5
            # Decrease bandwidth for better SNR
            waveform.bandwidth *= 0.7
            
        return waveform