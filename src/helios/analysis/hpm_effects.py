"""Analysis tools for HPM effects on electronic systems."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from helios.core.data_structures import HPMWaveform, Signal
from helios.environment.hpm_coupling import HPMEffect
from helios.effects.susceptibility import ComponentSusceptibility, EffectType
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EffectSummary:
    """Summary of HPM effects on a system."""
    total_effects: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, float] = field(default_factory=dict)  # Changed from None to field(default_factory=dict)
    by_component: Dict[str, int] = field(default_factory=dict)   # Changed from None to field(default_factory=dict)
    
    # The __post_init__ method is no longer needed since we're using field(default_factory=dict)
    # But we can keep it if you want to add additional initialization logic later

def analyze_hpm_effects(effects: List[HPMEffect]) -> EffectSummary:
    """
    Analyze a list of HPM effects and generate summary statistics.
    
    Args:
        effects: List of HPM effects
        
    Returns:
        Summary of effects
    """
    summary = EffectSummary(total_effects=len(effects))
    
    # Count effects by type
    for effect in effects:
        # Count by type
        effect_type = effect.effect_type
        summary.by_type[effect_type] = summary.by_type.get(effect_type, 0) + 1
        
        # Track by component
        component_id = effect.system_id
        summary.by_component[component_id] = summary.by_component.get(component_id, 0) + 1
        
        # Track average severity by type
        if effect_type not in summary.by_severity:
            summary.by_severity[effect_type] = effect.severity
        else:
            # Running average
            count = summary.by_type[effect_type]
            current_avg = summary.by_severity[effect_type]
            summary.by_severity[effect_type] = current_avg + (effect.severity - current_avg) / count
    
    return summary

def correlate_waveform_parameters_with_effects(
    waveforms: List[HPMWaveform], 
    effects_by_waveform: Dict[str, List[HPMEffect]]
) -> Dict[str, Any]:
    """
    Correlate waveform parameters with observed effects.
    
    Args:
        waveforms: List of HPM waveforms
        effects_by_waveform: Dictionary mapping waveform IDs to their effects
        
    Returns:
        Dictionary of correlations
    """
    correlations = {
        "frequency_vs_effects": [],
        "power_vs_effects": [],
        "pulse_width_vs_effects": [],
        "most_effective_waveforms": []
    }
    
    # Extract data points for correlation
    for waveform in waveforms:
        waveform_id = waveform.id
        if waveform_id not in effects_by_waveform:
            continue
            
        effects = effects_by_waveform[waveform_id]
        if not effects:
            continue
            
        # Count effects by severity
        effect_count = len(effects)
        damage_count = sum(1 for e in effects if e.effect_type in ["physical_damage", "functional_damage", "burnout"])
        upset_count = sum(1 for e in effects if e.effect_type in ["upset", "latch_up"])
        
        # Add data points
        correlations["frequency_vs_effects"].append((waveform.center_frequency, effect_count))
        correlations["power_vs_effects"].append((waveform.peak_power, effect_count))
        
        if hasattr(waveform, "pulse_width"):
            correlations["pulse_width_vs_effects"].append((waveform.pulse_width, effect_count))
        
        # Calculate effectiveness score (weighted sum of effects)
        effectiveness = damage_count * 3 + upset_count * 1
        
        correlations["most_effective_waveforms"].append({
            "waveform_id": waveform_id,
            "frequency": waveform.center_frequency,
            "peak_power": waveform.peak_power,
            "pulse_width": getattr(waveform, "pulse_width", None),
            "effectiveness": effectiveness,
            "damage_count": damage_count,
            "upset_count": upset_count
        })
    
    # Sort most effective waveforms
    correlations["most_effective_waveforms"].sort(key=lambda x: x["effectiveness"], reverse=True)
    
    return correlations

def predict_optimal_waveform_parameters(
    target_components: List[ComponentSusceptibility],
    frequency_range: Tuple[float, float],
    max_power: float
) -> Dict[str, Any]:
    """
    Predict optimal waveform parameters for affecting target components.
    
    Args:
        target_components: List of target component susceptibility models
        frequency_range: Tuple of (min_frequency, max_frequency) in Hz
        max_power: Maximum available power in Watts
        
    Returns:
        Dictionary of optimal parameters
    """
    # Simple implementation - find frequencies where components are most susceptible
    # Initialize with explicit typing to allow different value types
    optimal_params: Dict[str, Any] = {
        "frequency": None,
        "power": None,
        "pulse_width": None,
        "modulation": None
    }
    
    # Find frequencies where components have the lowest upset thresholds
    min_upset_threshold = float('inf')
    optimal_frequency = None
    
    for component in target_components:
        for effect_type, thresholds in component.frequency_thresholds.items():
            if effect_type.lower() != "upset":
                continue
                
            for freq, threshold in thresholds:
                if freq < frequency_range[0] or freq > frequency_range[1]:
                    continue
                    
                if threshold < min_upset_threshold:
                    min_upset_threshold = threshold
                    optimal_frequency = freq
    
    # If no specific frequency found, use middle of range
    if optimal_frequency is None:
        optimal_frequency = (frequency_range[0] + frequency_range[1]) / 2
    
    # Set optimal parameters
    optimal_params["frequency"] = optimal_frequency
    
    # Power should be enough to cause effects but not waste energy
    # Convert min_upset_threshold from dBm to Watts
    if min_upset_threshold != float('inf'):
        min_upset_power_watts = 10 ** ((min_upset_threshold - 30) / 10)
        # Add margin
        optimal_power = min(max_power, min_upset_power_watts * 2)
    else:
        optimal_power = max_power / 2  # Conservative default
    
    optimal_params["power"] = optimal_power
    
    # Short pulses are generally more effective for electronic upset
    optimal_params["pulse_width"] = 100e-9  # 100 ns default
    
    return optimal_params