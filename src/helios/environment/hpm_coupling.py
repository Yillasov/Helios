"""High-Power Microwave (HPM) coupling models."""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

from helios.core.data_structures import Position, Signal, Platform
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CouplingPath:
    """Represents an electromagnetic coupling path into a system."""
    name: str
    system_id: str  # ID of the system being affected
    frequency_range: Tuple[float, float]  # Min and max frequency in Hz
    coupling_coefficient: float  # dB
    threshold_power: float  # Minimum power in dBm to cause effects
    
    # Optional parameters for more detailed modeling
    polarization_factor: float = 1.0  # Multiplier for polarization match (0-1)
    angle_dependency: Optional[Dict[str, float]] = None  # Angle-dependent factors

@dataclass
class HPMEffect:
    """Represents the effect of HPM coupling on a system."""
    system_id: str
    effect_type: str  # e.g., "upset", "damage", "interference"
    severity: float  # 0-1 scale
    duration: float  # Effect duration in seconds
    description: str = ""

class HPMCouplingModel:
    """
    Models the coupling of high-power electromagnetic energy into electronic systems.
    Used to simulate electromagnetic interference and effects.
    """
    
    def __init__(self):
        """Initialize the HPM coupling model."""
        self.coupling_paths: Dict[str, List[CouplingPath]] = {}  # Platform ID -> coupling paths
        self.active_effects: List[Tuple[float, HPMEffect]] = []  # (end_time, effect)
    
    def add_coupling_path(self, platform_id: str, path: CouplingPath):
        """Add a coupling path to a platform."""
        if platform_id not in self.coupling_paths:
            self.coupling_paths[platform_id] = []
        
        self.coupling_paths[platform_id].append(path)
        logger.debug(f"Added coupling path '{path.name}' to platform {platform_id}")
    
    def calculate_coupling(self, signal: Signal, platform: Platform) -> List[HPMEffect]:
        """
        Calculate HPM coupling effects for a signal on a platform.
        
        Args:
            signal: The incident signal
            platform: The platform potentially affected
            
        Returns:
            List of HPM effects generated
        """
        if platform.id not in self.coupling_paths:
            return []
        
        effects = []
        signal_freq = signal.waveform.center_frequency
        
        for path in self.coupling_paths[platform.id]:
            # Check if signal frequency is in coupling path's range
            min_freq, max_freq = path.frequency_range
            if not (min_freq <= signal_freq <= max_freq):
                continue
            
            # Calculate coupled power
            coupled_power = signal.power + path.coupling_coefficient
            
            # Apply polarization factor
            if signal.polarization:
                # Simple polarization matching (can be enhanced)
                if signal.polarization == "vertical" and path.polarization_factor < 0.5:
                    coupled_power -= 20  # 20dB reduction for mismatched polarization
                elif signal.polarization == "horizontal" and path.polarization_factor > 0.5:
                    coupled_power -= 20
            
            # Apply angle dependency if available
            if path.angle_dependency and signal.direction:
                # Simplified angle dependency
                azimuth, elevation = signal.direction
                
                # Check if azimuth key exists and get the value
                if "azimuth" in path.angle_dependency:
                    azimuth_data = path.angle_dependency["azimuth"]
                    
                    # Handle different types of azimuth data
                    if isinstance(azimuth_data, dict):
                        # If it's a dictionary, look up the closest angle
                        angle_key = int(azimuth)
                        angle_factor = azimuth_data.get(angle_key, 0)
                    elif isinstance(azimuth_data, float):
                        # If it's a float, use it directly
                        angle_factor = azimuth_data
                    else:
                        angle_factor = 0
                        
                    coupled_power += angle_factor
            
            # Check if power exceeds threshold
            if coupled_power > path.threshold_power:
                # Calculate effect severity based on power above threshold
                power_ratio = 10**((coupled_power - path.threshold_power) / 10)
                severity = min(1.0, power_ratio / 100)  # Normalize to 0-1
                
                # Determine effect type and duration based on severity
                if severity > 0.8:
                    effect_type = "damage"
                    duration = float('inf')  # Permanent damage
                elif severity > 0.5:
                    effect_type = "upset"
                    duration = 30.0  # 30 seconds of system upset
                else:
                    effect_type = "interference"
                    duration = 5.0  # 5 seconds of interference
                
                # Create effect
                effect = HPMEffect(
                    system_id=path.system_id,
                    effect_type=effect_type,
                    severity=severity,
                    duration=duration,
                    description=f"HPM coupling via {path.name} at {coupled_power:.1f} dBm"
                )
                
                effects.append(effect)
                logger.info(f"HPM effect on platform {platform.id}, system {path.system_id}: "
                           f"{effect_type} (severity: {severity:.2f})")
        
        return effects
    
    def apply_effects(self, platform: Platform, effects: List[HPMEffect], current_time: float):
        """
        Apply HPM effects to a platform and schedule their end.
        
        Args:
            platform: The affected platform
            effects: List of effects to apply
            current_time: Current simulation time
        """
        for effect in effects:
            # Apply effect to the system
            if effect.system_id in platform.equipped_systems:
                system = platform.equipped_systems[effect.system_id]
                
                # Store effect information in system parameters
                if "hpm_effects" not in system.parameters:
                    system.parameters["hpm_effects"] = []
                
                system.parameters["hpm_effects"].append({
                    "type": effect.effect_type,
                    "severity": effect.severity,
                    "start_time": current_time,
                    "description": effect.description
                })
                
                # For damage effects, mark system as damaged
                if effect.effect_type == "damage":
                    system.parameters["damaged"] = True
                    logger.warning(f"System {effect.system_id} on platform {platform.id} damaged by HPM")
            
            # Schedule effect end if not permanent
            if effect.duration < float('inf'):
                end_time = current_time + effect.duration
                self.active_effects.append((end_time, effect))
    
    def update_effects(self, current_time: float, platforms: Dict[str, Platform]):
        """
        Update active effects, removing expired ones.
        
        Args:
            current_time: Current simulation time
            platforms: Dictionary of all platforms
        """
        # Remove expired effects
        active_effects = []
        for end_time, effect in self.active_effects:
            if current_time >= end_time:
                # Effect has expired, remove it from the system
                for platform in platforms.values():
                    if effect.system_id in platform.equipped_systems:
                        system = platform.equipped_systems[effect.system_id]
                        if "hpm_effects" in system.parameters:
                            # Remove this specific effect
                            system.parameters["hpm_effects"] = [
                                e for e in system.parameters["hpm_effects"]
                                if e["type"] != effect.effect_type or e["severity"] != effect.severity
                            ]
                            
                            logger.debug(f"HPM effect on system {effect.system_id} expired")
            else:
                active_effects.append((end_time, effect))
        
        self.active_effects = active_effects