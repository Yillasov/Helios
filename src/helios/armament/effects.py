"""Effects modeling for RF-based armament systems."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

from helios.core.data_structures import Waveform, Signal, HPMWaveform
from helios.effects.rf_effects import RFEffect, RFEffectType, RFEffectSeverity
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class EffectCategory(Enum):
    """Categories of effects from RF-based armament systems."""
    ELECTRONIC = auto()
    COMMUNICATIONS = auto()
    SENSOR = auto()
    CONTROL = auto()
    INFRASTRUCTURE = auto()


@dataclass
class EffectAssessment:
    """Assessment of effects on a target system."""
    target_id: str
    effect_probability: float
    effect_category: EffectCategory
    estimated_duration: float  # seconds
    confidence: float  # 0-1 scale


class EffectsModel:
    """Models effects of RF-based armament systems on targets."""
    
    def __init__(self):
        """Initialize the effects model."""
        self.effect_database: Dict[str, Dict[str, Any]] = {}
        
    def predict_effects(self, 
                       waveform: Waveform, 
                       target_vulnerability: Dict[str, Any],
                       distance: float) -> List[EffectAssessment]:
        """
        Predict effects of a waveform on a target.
        
        Args:
            waveform: RF waveform characteristics
            target_vulnerability: Target vulnerability profile
            distance: Distance to target in meters
            
        Returns:
            List of effect assessments
        """
        assessments = []
        
        # Calculate power at target (simplified free space path loss)
        frequency = waveform.center_frequency
        wavelength = 3e8 / frequency  # c/f
        
        # Get power from waveform - handle different waveform types
        if isinstance(waveform, HPMWaveform):
            power = waveform.peak_power
        else:
            # For standard waveforms, estimate power from amplitude (assuming 50 ohm impedance)
            power = (waveform.amplitude ** 2) / 50
            # Convert to dBm
            power = 10 * np.log10(power * 1000)
        
        # Free space path loss
        path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
        power_at_target = power - path_loss_db
        
        # Assess effects on different systems
        for system_id, vulnerability in target_vulnerability.items():
            # Skip if system not vulnerable to this frequency
            if not (vulnerability.get("min_freq", 0) <= frequency <= vulnerability.get("max_freq", 1e12)):
                continue
                
            # Calculate effect probability based on power and vulnerability
            threshold = vulnerability.get("effect_threshold", 0)
            power_margin = power_at_target - threshold
            
            if power_margin <= 0:
                continue  # Below threshold
                
            # Simple probability model
            probability = min(0.95, power_margin / 20)  # Cap at 95%
            
            # Create assessment
            assessment = EffectAssessment(
                target_id=system_id,
                effect_probability=probability,
                effect_category=EffectCategory[vulnerability.get("category", "ELECTRONIC")],
                estimated_duration=vulnerability.get("effect_duration", 60),
                confidence=0.7  # Default confidence
            )
            
            assessments.append(assessment)
            
        return assessments


class CollateralAnalyzer:
    """Analyzes potential collateral effects of RF-based armament systems."""
    
    def __init__(self):
        """Initialize the collateral analyzer."""
        self.civilian_infrastructure: Dict[str, Dict[str, Any]] = {}
        self.effects_model = EffectsModel()
        
    def analyze_collateral_risk(self,
                              waveform: Waveform,
                              target_position: Tuple[float, float, float],
                              infrastructure_positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Analyze risk of collateral effects on civilian infrastructure.
        
        Args:
            waveform: RF waveform characteristics
            target_position: Position of intended target (x, y, z) in meters
            infrastructure_positions: Dictionary mapping infrastructure IDs to positions
            
        Returns:
            Dictionary mapping infrastructure IDs to risk scores (0-1)
        """
        risks = {}
        
        for infra_id, position in infrastructure_positions.items():
            # Calculate distance
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(target_position, position)))
            
            # Skip if too far
            if distance > 5000:  # 5km
                continue
                
            # Get vulnerability profile
            vulnerability = self.civilian_infrastructure.get(infra_id, {
                "category": "INFRASTRUCTURE",
                "effect_threshold": 30,  # dBm
                "min_freq": 100e6,
                "max_freq": 10e9,
                "effect_duration": 300  # seconds
            })
            
            # Predict effects
            assessments = self.effects_model.predict_effects(
                waveform=waveform,
                target_vulnerability={infra_id: vulnerability},
                distance=distance
            )
            
            # Calculate risk based on effects
            if assessments:
                risk = assessments[0].effect_probability * 0.8  # Scale factor
                risks[infra_id] = risk
            
        return risks