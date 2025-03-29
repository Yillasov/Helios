"""Adaptive targeting modules based on environmental conditions."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import time

from helios.core.data_structures import Signal, Position, EnvironmentParameters
from helios.targets.rf_signature_db import RFSignature
from helios.targets.recognition import TargetRecognitionAlgorithm
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TargetPriority:
    """Target priority information."""
    target_id: str
    priority_score: float = 1.0  # Higher is more important
    confidence: float = 1.0  # Recognition confidence
    last_updated: float = field(default_factory=time.time)

class AdaptiveTargetingSystem:
    """System for adapting targeting based on environmental conditions."""
    
    def __init__(self, recognition_algorithm: TargetRecognitionAlgorithm):
        """Initialize with a recognition algorithm."""
        self.recognition_algorithm = recognition_algorithm
        self.target_priorities: Dict[str, TargetPriority] = {}
        self.environment_params: Optional[EnvironmentParameters] = None
        self.weather_conditions: Dict[str, Any] = {}
        
    def set_environment_parameters(self, params: EnvironmentParameters):
        """Set current environment parameters."""
        self.environment_params = params
        
    def set_weather_conditions(self, conditions: Dict[str, Any]):
        """Set current weather conditions."""
        self.weather_conditions = conditions
        
    def process_signal(self, signal: Signal) -> List[Tuple[RFSignature, float]]:
        """
        Process a detected signal and update target priorities.
        
        Args:
            signal: Detected RF signal
            
        Returns:
            List of (signature, adjusted_priority) tuples
        """
        # Recognize the target
        matches = self.recognition_algorithm.recognize(signal)
        
        # Update priorities based on recognition and environment
        results = []
        for signature, confidence in matches:
            # Calculate base priority
            priority = self._calculate_base_priority(signature)
            
            # Apply environmental adjustments
            adjusted_priority = self._adjust_for_environment(signature, priority)
            
            # Store updated priority
            self.target_priorities[signature.id] = TargetPriority(
                target_id=signature.id,
                priority_score=adjusted_priority,
                confidence=confidence,
                last_updated=time.time()
            )
            
            results.append((signature, adjusted_priority))
            
        # Sort by adjusted priority, highest first
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _calculate_base_priority(self, signature: RFSignature) -> float:
        """Calculate base priority for a target."""
        # Simple implementation - can be extended with more complex logic
        priority = 1.0
        
        # Prioritize based on signature type
        if signature.signature_type.name == "ELECTRONIC_WARFARE":
            priority *= 2.0
        elif signature.signature_type.name == "RADAR":
            priority *= 1.5
            
        return priority
    
    def _adjust_for_environment(self, signature: RFSignature, base_priority: float) -> float:
        """Adjust priority based on environmental conditions."""
        adjusted_priority = base_priority
        
        # No adjustment if no environment data
        if not self.environment_params:
            return adjusted_priority
            
        # Adjust for temperature effects on electronics
        if self.environment_params.temperature > 310:  # Hot environment (>37°C)
            # Electronic systems may be more vulnerable in hot conditions
            adjusted_priority *= 1.2
            
        # Adjust for weather conditions
        if "precipitation" in self.weather_conditions:
            precip = self.weather_conditions["precipitation"]
            if precip > 5.0:  # Heavy rain (mm/hour)
                # Radar and communication systems may be degraded in heavy rain
                if signature.signature_type.name in ["RADAR", "COMMUNICATION"]:
                    # Higher frequency systems are more affected by rain
                    if signature.features.center_frequency > 10e9:  # Above 10 GHz
                        adjusted_priority *= 0.8
        
        # Adjust for time of day if available
        if "time_of_day" in self.weather_conditions:
            time_of_day = self.weather_conditions["time_of_day"]
            if time_of_day == "night":
                # Some systems may be more active or important at night
                if "night_operation" in signature.features.additional_features:
                    adjusted_priority *= 1.3
                    
        return adjusted_priority
    
    def get_prioritized_targets(self, max_targets: int = 10) -> List[Tuple[RFSignature, float]]:
        """
        Get a list of prioritized targets.
        
        Args:
            max_targets: Maximum number of targets to return
            
        Returns:
            List of (signature, priority) tuples, sorted by priority
        """
        # Get signatures for all priorities
        results = []
        for target_id, priority in self.target_priorities.items():
            signature = self.recognition_algorithm.signature_db.get_signature(target_id)
            if signature:
                results.append((signature, priority.priority_score))
                
        # Sort by priority, highest first
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_targets]