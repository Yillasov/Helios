"""Cascading effects simulation for complex systems."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import time
import numpy as np

from helios.core.data_structures import Platform, System
from helios.effects.system_effects import SystemEffect, SystemEffectType, SystemEffectsPredictor
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CascadingEffect:
    """Represents a cascading effect between systems."""
    source_system_id: str
    target_system_id: str
    propagation_delay: float  # Time delay before effect propagates (seconds)
    effect_type: SystemEffectType
    severity_factor: float = 0.8  # How much of the original severity propagates (0-1)
    description: str = ""
    probability: float = 1.0  # Probability of cascade occurring (0-1)

class CascadingEffectsSimulator:
    """Simulates cascading effects between interconnected systems."""
    
    def __init__(self, system_effects_predictor: SystemEffectsPredictor):
        """Initialize the cascading effects simulator."""
        self.system_effects_predictor = system_effects_predictor
        self.cascade_rules: Dict[str, List[CascadingEffect]] = {}  # system_id -> list of potential cascades
        self.pending_cascades: List[Tuple[float, SystemEffect]] = []  # (trigger_time, effect)
        self.last_update_time: float = 0.0
    
    def add_cascade_rule(self, cascade: CascadingEffect):
        """Add a cascade rule between systems."""
        if cascade.source_system_id not in self.cascade_rules:
            self.cascade_rules[cascade.source_system_id] = []
        self.cascade_rules[cascade.source_system_id].append(cascade)
        logger.debug(f"Added cascade rule: {cascade.source_system_id} -> {cascade.target_system_id}")
    
    def process_system_effects(self, effects: List[SystemEffect], current_time: float) -> List[SystemEffect]:
        """
        Process system effects and identify potential cascading effects.
        
        Args:
            effects: List of system effects to process
            current_time: Current simulation time
            
        Returns:
            List of additional effects from cascades triggered immediately
        """
        self.last_update_time = current_time
        immediate_cascades = []
        
        # Process each effect for potential cascades
        for effect in effects:
            if effect.system_id in self.cascade_rules:
                for cascade in self.cascade_rules[effect.system_id]:
                    # Check if cascade should trigger (based on probability)
                    if np.random.random() <= cascade.probability:
                        # Calculate when cascade will trigger
                        trigger_time = current_time + cascade.propagation_delay
                        
                        # Create the cascaded effect
                        cascaded_effect = SystemEffect(
                            system_id=cascade.target_system_id,
                            effect_type=cascade.effect_type,
                            severity=effect.severity * cascade.severity_factor,
                            duration=effect.duration * 0.8,  # Cascaded effects typically shorter
                            affected_components=[],  # Will be populated when applied
                            description=f"Cascade from {effect.system_id}: {cascade.description or effect.description}",
                            recovery_time=effect.recovery_time * 1.2  # Cascaded effects take longer to recover
                        )
                        
                        # If delay is zero, add to immediate effects
                        if cascade.propagation_delay <= 0.001:  # Small threshold for "immediate"
                            immediate_cascades.append(cascaded_effect)
                            logger.info(f"Immediate cascade: {effect.system_id} -> {cascade.target_system_id}")
                        else:
                            # Otherwise schedule for future processing
                            self.pending_cascades.append((trigger_time, cascaded_effect))
                            logger.info(f"Scheduled cascade at t={trigger_time:.2f}s: {effect.system_id} -> {cascade.target_system_id}")
        
        return immediate_cascades
    
    def update(self, current_time: float) -> List[SystemEffect]:
        """
        Update cascading effects simulation, triggering any pending cascades.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of new effects triggered by cascades
        """
        if current_time <= self.last_update_time:
            return []
            
        self.last_update_time = current_time
        triggered_effects = []
        still_pending = []
        
        # Check for cascades that should trigger now
        for trigger_time, effect in self.pending_cascades:
            if current_time >= trigger_time:
                triggered_effects.append(effect)
                logger.info(f"Triggered cascade at t={current_time:.2f}s: {effect.system_id}")
            else:
                still_pending.append((trigger_time, effect))
        
        # Update pending cascades list
        self.pending_cascades = still_pending
        
        return triggered_effects
    
    def apply_cascading_effects(self, platform: Platform, current_time: float):
        """
        Apply any triggered cascading effects to a platform.
        
        Args:
            platform: Platform to apply effects to
            current_time: Current simulation time
        """
        # Get effects that should trigger at this time
        effects = self.update(current_time)
        
        if effects:
            # Apply the effects to the platform
            self.system_effects_predictor.apply_system_effects(platform, effects, current_time)
            
            # Process these effects for further cascades
            secondary_effects = self.process_system_effects(effects, current_time)
            
            # Apply any immediate secondary cascades
            if secondary_effects:
                self.system_effects_predictor.apply_system_effects(platform, secondary_effects, current_time)