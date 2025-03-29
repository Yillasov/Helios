"""High-Power Microwave (HPM) coupling models."""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union
from dataclasses import dataclass, field
import time

from helios.core.data_structures import Position, Signal, Platform, HPMWaveform, PulsedWaveform
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ThermalProperties:
    """Thermal properties of a system or component."""
    thermal_conductivity: float = 0.5  # W/(m·K)
    specific_heat_capacity: float = 800.0  # J/(kg·K)
    mass: float = 1.0  # kg
    surface_area: float = 0.1  # m²
    max_safe_temperature: float = 85.0  # °C
    critical_temperature: float = 125.0  # °C
    ambient_temperature: float = 25.0  # °C
    cooling_coefficient: float = 0.05  # Heat dissipation rate

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
    thermal_properties: Optional[ThermalProperties] = None  # Thermal properties

@dataclass
class HPMEffect:
    """Represents the effect of HPM coupling on a system."""
    system_id: str
    effect_type: str  # e.g., "upset", "damage", "interference", "thermal"
    severity: float  # 0-1 scale
    duration: float  # Effect duration in seconds
    description: str = ""
    temperature_rise: float = 0.0  # Temperature increase in °C

class ThermalEffectsSimulator:
    """Simulates thermal effects from HPM energy absorption."""
    
    def __init__(self):
        """Initialize the thermal effects simulator."""
        self.system_temperatures: Dict[str, float] = {}  # Current temperature of each system
        self.last_update_time: float = 0.0
    
    def calculate_temperature_rise(self, 
                                  coupled_power_dbm: float, 
                                  exposure_time: float,
                                  thermal_props: ThermalProperties) -> float:
        """
        Calculate temperature rise from absorbed HPM energy.
        
        Args:
            coupled_power_dbm: Power coupled into system in dBm
            exposure_time: Duration of exposure in seconds
            thermal_props: Thermal properties of the system
            
        Returns:
            Temperature rise in degrees Celsius
        """
        # Convert dBm to Watts
        power_watts = 10 ** ((coupled_power_dbm - 30) / 10)
        
        # Calculate energy absorbed (Joules)
        energy_absorbed = power_watts * exposure_time
        
        # Calculate temperature rise (ΔT = Q / (m * c))
        # Q = energy, m = mass, c = specific heat capacity
        temp_rise = energy_absorbed / (thermal_props.mass * thermal_props.specific_heat_capacity)
        
        return temp_rise
    
    def update_temperatures(self, current_time: float, systems_dict: Dict[str, Any]):
        """
        Update temperatures of all systems based on cooling over time.
        
        Args:
            current_time: Current simulation time
            systems_dict: Dictionary of systems to update
        """
        if current_time <= self.last_update_time:
            return
            
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update each system's temperature
        for system_id, temp in list(self.system_temperatures.items()):
            if system_id in systems_dict:
                system = systems_dict[system_id]
                
                # Get thermal properties
                thermal_props = None
                if hasattr(system, "thermal_properties"):
                    thermal_props = system.thermal_properties
                else:
                    # Use default properties
                    thermal_props = ThermalProperties()
                
                # Calculate cooling (Newton's law of cooling)
                ambient_temp = thermal_props.ambient_temperature
                cooling_rate = thermal_props.cooling_coefficient
                
                # T(t) = Ta + (T0 - Ta) * e^(-kt)
                # where Ta = ambient temp, T0 = initial temp, k = cooling coefficient, t = time
                new_temp = ambient_temp + (temp - ambient_temp) * np.exp(-cooling_rate * time_delta)
                
                # Update temperature
                self.system_temperatures[system_id] = new_temp
                
                # Store temperature in system parameters
                if hasattr(system, "parameters"):
                    system.parameters["temperature"] = new_temp
                    
                    # Check for thermal recovery
                    if new_temp < thermal_props.max_safe_temperature and "thermal_effect" in system.parameters:
                        system.parameters.pop("thermal_effect")
                        logger.info(f"System {system_id} recovered from thermal effects at {new_temp:.1f}°C")

# Update the HPMCouplingModel class to include thermal effects
class HPMCouplingModel:
    """
    Models the coupling of high-power electromagnetic energy into electronic systems.
    Used to simulate electromagnetic interference and effects.
    """
    
    def __init__(self):
        """Initialize the HPM coupling model."""
        self.coupling_paths: Dict[str, List[CouplingPath]] = {}  # Platform ID -> coupling paths
        self.active_effects: List[Tuple[float, HPMEffect]] = []  # (end_time, effect)
        self.thermal_simulator = ThermalEffectsSimulator()
    
    def add_coupling_path(self, platform_id: str, path: CouplingPath):
        """Add a coupling path to a platform."""
        if platform_id not in self.coupling_paths:
            self.coupling_paths[platform_id] = []
        
        self.coupling_paths[platform_id].append(path)
        logger.debug(f"Added coupling path '{path.name}' to platform {platform_id}")
    
    def calculate_detailed_coupling(self, signal: Signal, platform: Platform) -> List[HPMEffect]:
        """
        Calculate HPM coupling effects with more detailed models.
        
        Args:
            signal: The incident signal
            platform: The platform potentially affected
            
        Returns:
            List of HPM effects generated
        """
        if platform.id not in self.coupling_paths:
            return []
        
        effects = []
        
        # Check if signal has HPM waveform
        if not hasattr(signal.waveform, 'center_frequency'):
            logger.debug(f"Signal {signal.id} doesn't have required attributes")
            return []
            
        signal_freq = signal.waveform.center_frequency
        
        # Get pulse parameters if available (for HPM waveforms)
        pulse_width = None
        peak_power = None
        if isinstance(signal.waveform, HPMWaveform):
            pulse_width = signal.waveform.pulse_width
            peak_power = signal.waveform.peak_power
        
        for path in self.coupling_paths[platform.id]:
            # Check if signal frequency is in coupling path's range
            min_freq, max_freq = path.frequency_range
            if not (min_freq <= signal_freq <= max_freq):
                continue
            
            # Calculate base coupled power
            coupled_power = signal.power + path.coupling_coefficient
            
            # Apply frequency-dependent effects
            # Simple model: coupling is strongest at center of frequency range
            freq_range_center = (min_freq + max_freq) / 2
            freq_range_width = max_freq - min_freq
            
            # Reduce coupling as we move away from center frequency
            if freq_range_width > 0:
                freq_factor = 1.0 - min(1.0, 2.0 * abs(signal_freq - freq_range_center) / freq_range_width)
                # Apply frequency-dependent attenuation (up to 10dB)
                coupled_power += 10 * (freq_factor - 0.5)
            
            # Apply pulse width effects for HPM signals
            if pulse_width is not None:
                # Short pulses couple differently than long pulses
                if pulse_width < 1e-6:  # Less than 1 microsecond
                    coupled_power += 3  # Short pulses may couple better in some cases
                elif pulse_width > 1e-3:  # Greater than 1 millisecond
                    coupled_power -= 3  # Long pulses may couple worse
            
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
            
            # Calculate thermal effects if thermal properties are available
            thermal_props = path.thermal_properties
            temp_rise = 0.0
            
            if thermal_props and coupled_power > (path.threshold_power - 10):  # Even below threshold, heat can build up
                # Calculate temperature rise based on coupled power
                # Assume 100ms exposure for each signal
                exposure_time = 0.1
                if pulse_width is not None:
                    # For pulsed signals, consider duty cycle
                    if isinstance(signal.waveform, PulsedWaveform):
                        exposure_time = pulse_width * signal.waveform.duty_cycle * 10  # Assume 10 pulses
                    else:
                        exposure_time = pulse_width * 10  # Assume 10 pulses
                
                temp_rise = self.thermal_simulator.calculate_temperature_rise(
                    coupled_power, exposure_time, thermal_props)
                
                # Get current temperature or use ambient
                system_id = path.system_id
                current_temp = self.thermal_simulator.system_temperatures.get(
                    system_id, thermal_props.ambient_temperature)
                
                # Calculate new temperature
                new_temp = current_temp + temp_rise
                self.thermal_simulator.system_temperatures[system_id] = new_temp
                
                # Check if temperature exceeds thresholds
                if new_temp > thermal_props.critical_temperature:
                    # Critical temperature - permanent damage
                    thermal_effect = HPMEffect(
                        system_id=path.system_id,
                        effect_type="thermal_damage",
                        severity=1.0,
                        duration=float('inf'),  # Permanent
                        description=f"Thermal damage at {new_temp:.1f}°C (critical: {thermal_props.critical_temperature}°C)",
                        temperature_rise=temp_rise
                    )
                    effects.append(thermal_effect)
                    
                elif new_temp > thermal_props.max_safe_temperature:
                    # Exceeds safe temperature - temporary effect
                    # Duration based on how much it exceeds safe temp
                    excess_temp = new_temp - thermal_props.max_safe_temperature
                    duration = min(300, excess_temp * 10)  # 10 seconds per degree, max 5 minutes
                    
                    thermal_effect = HPMEffect(
                        system_id=path.system_id,
                        effect_type="thermal_degradation",
                        severity=min(1.0, excess_temp / (thermal_props.critical_temperature - thermal_props.max_safe_temperature)),
                        duration=duration,
                        description=f"Thermal degradation at {new_temp:.1f}°C (safe: {thermal_props.max_safe_temperature}°C)",
                        temperature_rise=temp_rise
                    )
                    effects.append(thermal_effect)
            
            # Check if power exceeds threshold for direct effects
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
                    description=f"HPM effect via {path.name} coupling path",
                    temperature_rise=temp_rise
                )
                
                effects.append(effect)
        
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
                
                effect_info = {
                    "type": effect.effect_type,
                    "severity": effect.severity,
                    "start_time": current_time,
                    "description": effect.description
                }
                
                # Add thermal information if present
                if effect.temperature_rise > 0:
                    effect_info["temperature_rise"] = effect.temperature_rise
                    system.parameters["temperature"] = self.thermal_simulator.system_temperatures.get(
                        effect.system_id, 25.0)  # Default to room temp if not set
                    
                    # Mark thermal effect
                    if "thermal" in effect.effect_type:
                        system.parameters["thermal_effect"] = True
                
                system.parameters["hpm_effects"].append(effect_info)
                
                # For damage effects, mark system as damaged
                if effect.effect_type == "damage" or effect.effect_type == "thermal_damage":
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
        # Update thermal simulation
        systems_dict = {}
        for platform in platforms.values():
            for system_id, system in platform.equipped_systems.items():
                systems_dict[system_id] = system
                
        self.thermal_simulator.update_temperatures(current_time, systems_dict)
        
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