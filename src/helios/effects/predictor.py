"""Prediction models for HPM effects on electronic components."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from helios.core.data_structures import Signal, HPMWaveform
from helios.environment.hpm_coupling import HPMEffect, CouplingPath
from helios.effects.susceptibility import ComponentSusceptibility, EffectType
# Assuming ModulationType is defined elsewhere
# from helios.core.data_structures import ModulationType 
from helios.effects.effects_database import EffectsDatabase, EffectRecord
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class HPMEffectsPredictor:
    """Predicts effects of HPM signals on electronic components."""
    
    def __init__(self, use_database: bool = True, db_path: Optional[str] = None):
        """Initialize the HPM effects predictor."""
        self.component_models: Dict[str, ComponentSusceptibility] = {}
        self.use_database = use_database
        self.database = EffectsDatabase(db_path) if use_database else None
        logger.info(f"Initialized HPM effects predictor with database: {use_database}")
    
    def add_component_model(self, component: ComponentSusceptibility):
        """Add a component susceptibility model."""
        self.component_models[component.component_id] = component
        logger.debug(f"Added susceptibility model for component {component.component_id}")
    
    def predict_effect(self, 
                      component_id: str, 
                      coupled_power: float, 
                      frequency: float,
                      pulse_width: Optional[float] = None,
                      # Add modulation_type parameter (can be ModulationType enum or string)
                      modulation_type: Optional[Any] = None) -> Tuple[EffectType, float]: 
        """
        Predict the effect on a component based on coupled power and frequency.
        
        Args:
            component_id: ID of the component
            coupled_power: Power coupled into the component (dBm)
            frequency: Signal frequency (Hz)
            pulse_width: Signal pulse width (seconds), if applicable
            modulation_type: Type of modulation (e.g., ModulationType.AM or "AM"), if applicable
            
        Returns:
            Tuple of (effect_type, severity)
        """
        if component_id not in self.component_models:
            logger.warning(f"No susceptibility model for component {component_id}")
            return EffectType.NONE, 0.0
        
        component = self.component_models[component_id]
        
        # Check frequency-specific thresholds first
        freq_specific_upset = float('inf')
        freq_specific_damage = float('inf')
        
        for effect_type, thresholds in component.frequency_thresholds.items():
            for freq_range, threshold in thresholds:
                if abs(frequency - freq_range) <= 0.1 * freq_range:  # Within 10% of specified frequency
                    if effect_type.lower() == "upset":
                        freq_specific_upset = threshold
                    elif effect_type.lower() in ["damage", "burnout"]:
                        freq_specific_damage = threshold
        
        # Use frequency-specific thresholds if available, otherwise use defaults
        # (Assuming upset_threshold & damage_threshold are determined as before)
        upset_threshold = freq_specific_upset if freq_specific_upset != float('inf') else component.upset_threshold
        damage_threshold = freq_specific_damage if freq_specific_damage != float('inf') else component.damage_threshold

        # Apply modulation factor
        mod_factor = 1.0
        if modulation_type:
            # Get the name of the modulation type (handle enum or string)
            mod_name = modulation_type.name if hasattr(modulation_type, 'name') else str(modulation_type)
            mod_factor = component.modulation_factors.get(mod_name, 1.0)
            if mod_factor != 1.0:
                 logger.debug(f"Applying modulation factor {mod_factor} for {mod_name} to component {component_id}")

        effective_upset_threshold = upset_threshold + 10 * np.log10(mod_factor) # Adjust dB threshold based on factor
        effective_damage_threshold = damage_threshold + 10 * np.log10(mod_factor) # Adjust dB threshold based on factor

        # Apply pulse width sensitivity (assuming calculation happens here)
        # Simplified: Adjust thresholds based on pulse width sensitivity factor
        pw_upset_factor = component.pulse_width_sensitivity.get("upset", 1.0)
        pw_damage_factor = component.pulse_width_sensitivity.get("damage", 1.0)

        # Example simple adjustment (could be more complex, e.g., energy-based)
        if pulse_width and pulse_width < 1e-6: # Example: less sensitive to very short pulses
             effective_upset_threshold /= (pw_upset_factor * (1e-6 / pulse_width)**0.5) # Threshold increases for short pulses
             effective_damage_threshold /= (pw_damage_factor * (1e-6 / pulse_width)**0.5) 

        # Determine effect based on adjusted thresholds
        if coupled_power >= effective_damage_threshold:
            severity = np.clip((coupled_power - effective_damage_threshold) / 10.0, 0, 1) # Scale severity
            # Check for specific damage types if needed
            return EffectType.BURNOUT, severity # Simplified to Burnout
        elif coupled_power >= effective_upset_threshold:
            severity = np.clip((coupled_power - effective_upset_threshold) / (effective_damage_threshold - effective_upset_threshold), 0, 1)
            # Check for specific upset types if needed
            return EffectType.UPSET, severity # Simplified to Upset
        else:
            # Consider interference threshold if defined
            return EffectType.NONE, 0.0

    def get_recovery_time(self, component_id: str, effect_type: EffectType) -> float:
        """Get the recovery time for a given effect on a component."""
        if component_id not in self.component_models:
            return 0.0
        
        component = self.component_models[component_id]
        return component.recovery_time.get(effect_type, 0.0)
    
    # In the predict_effects_from_signal method:
    def predict_effects_from_signal(self, 
                                   signal: Signal, 
                                   coupling_paths: List[CouplingPath]) -> List[HPMEffect]:
        """
        Predict effects from a signal using coupling paths.
        
        Args:
            signal: The HPM signal
            coupling_paths: List of coupling paths to components
            
        Returns:
            List of predicted HPM effects
        """
        effects = []
        
        # Extract waveform parameters
        if not isinstance(signal.waveform, HPMWaveform):
            logger.warning("Signal does not contain an HPM waveform")
            return effects
        
        waveform = signal.waveform
        frequency = waveform.center_frequency
        pulse_width = getattr(waveform, 'pulse_width', None)
        modulation_type = getattr(waveform, 'modulation_type', None)
        
        for path in coupling_paths:
            # Check if frequency is in coupling path's range
            min_freq, max_freq = path.frequency_range
            if not (min_freq <= frequency <= max_freq):
                continue
            
            # Calculate coupled power
            coupled_power = signal.power + path.coupling_coefficient
            
            # Skip if below threshold
            if coupled_power < path.threshold_power:
                continue
            
            # Get component ID from the coupling path
            component_id = path.system_id
            
            # Predict effect
            effect_type, severity = self.predict_effect(
                component_id, coupled_power, frequency, pulse_width, modulation_type
            )
            
            if effect_type != EffectType.NONE:
                # Determine effect duration based on component recovery time
                if component_id in self.component_models:
                    component = self.component_models[component_id]
                    duration = component.recovery_time.get(effect_type, 1.0)
                else:
                    # Default durations if component model not available
                    if effect_type in [EffectType.PHYSICAL_DAMAGE, EffectType.FUNCTIONAL_DAMAGE, EffectType.BURNOUT]:
                        duration = float('inf')  # Permanent damage
                    elif effect_type == EffectType.LATCH_UP:
                        duration = 10.0  # 10 seconds
                    elif effect_type == EffectType.UPSET:
                        duration = 1.0   # 1 second
                    else:
                        duration = 0.1   # 100ms
                
                # Create HPM effect
                hpm_effect = HPMEffect(
                    system_id=component_id,
                    effect_type=effect_type.name.lower(),
                    severity=severity,
                    duration=duration,
                    description=f"HPM effect from {waveform.id} at {frequency/1e6:.2f} MHz"
                )
                
                effects.append(hpm_effect)
                
                # Store in database if enabled
                if self.use_database and self.database and component_id in self.component_models:
                    # Get distance from path.angle_dependency if available, otherwise None
                    distance = None
                    if hasattr(path, 'angle_dependency') and path.angle_dependency and 'distance' in path.angle_dependency:
                        distance = path.angle_dependency.get('distance')
                    
                    self.database.add_effect_from_simulation(
                        effect=hpm_effect,
                        waveform=waveform,
                        target=self.component_models[component_id],
                        distance=distance,  # Use the extracted distance or None
                        measured=False  # This is a prediction
                    )
        
        return effects
    
    # Add new method to record measured effects
    def record_measured_effect(self, 
                              effect: HPMEffect, 
                              waveform: HPMWaveform,
                              target_id: str,
                              distance: Optional[float] = None):
        """
        Record a measured effect in the database.
        
        Args:
            effect: The measured HPM effect
            waveform: The waveform that caused the effect
            target_id: ID of the affected component
            distance: Distance between source and target (meters)
        """
        if not self.use_database or not self.database:
            logger.warning("Database not enabled, cannot record measured effect")
            return
        
        if target_id not in self.component_models:
            logger.warning(f"No susceptibility model for component {target_id}")
            return
        
        self.database.add_effect_from_simulation(
            effect=effect,
            waveform=waveform,
            target=self.component_models[target_id],
            distance=distance,  # This is already an optional parameter
            measured=True  # This is a measurement
        )
        logger.info(f"Recorded measured effect on {target_id} from waveform {waveform.id}")