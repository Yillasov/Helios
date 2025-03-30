"""Jamming effectiveness prediction algorithms for electronic warfare applications."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from helios.utils.logger import get_logger
from helios.ecm.techniques import ECMTechniqueType, ECMParameters
from helios.core.data_structures import Position, Signal

logger = get_logger(__name__)

class JammerType(Enum):
    """Types of jammers based on their operational characteristics."""
    BARRAGE = auto()       # Wide bandwidth noise jamming
    SPOT = auto()          # Narrow bandwidth, high power jamming
    SWEEP = auto()         # Frequency sweeping jamming
    DECEPTIVE = auto()     # False target generation
    REACTIVE = auto()      # Triggered by detected signals
    STANDOFF = auto()      # Long-range jamming
    ESCORT = auto()        # Accompanies strike package
    SELF_PROTECTION = auto() # Carried by protected platform


@dataclass
class JammingScenario:
    """Parameters describing a jamming scenario."""
    jammer_position: Position
    target_position: Position
    jammer_type: JammerType
    jammer_power: float  # Watts ERP
    jammer_frequency: float  # Hz
    jammer_bandwidth: float  # Hz
    target_frequency: float  # Hz
    target_bandwidth: float  # Hz
    target_sensitivity: float = -100.0  # dBm
    terrain_loss: float = 0.0  # dB
    atmospheric_loss: float = 0.0  # dB


@dataclass
class JammingEffectiveness:
    """Results of jamming effectiveness prediction."""
    jamming_to_signal_ratio: float  # J/S ratio in dB
    probability_of_denial: float  # 0-1 scale
    effective_range: float  # meters
    burn_through_range: float  # meters
    frequency_coverage: float  # 0-1 scale
    vulnerability_score: float  # 0-1 scale
    countermeasure_resistance: float  # 0-1 scale
    overall_effectiveness: float  # 0-1 scale


class JammingPredictor:
    """Predicts the effectiveness of electronic jamming in various scenarios."""
    
    def __init__(self):
        """Initialize the jamming predictor."""
        self.scenarios: Dict[str, JammingScenario] = {}
        self.results: Dict[str, JammingEffectiveness] = {}
        
    def create_scenario(self, 
                       scenario_id: str,
                       jammer_position: Position,
                       target_position: Position,
                       jammer_type: JammerType,
                       jammer_power: float,
                       jammer_frequency: float,
                       jammer_bandwidth: float,
                       target_frequency: float,
                       target_bandwidth: float,
                       target_sensitivity: float = -100.0,
                       terrain_loss: float = 0.0,
                       atmospheric_loss: float = 0.0) -> JammingScenario:
        """Create a jamming scenario for analysis.
        
        Args:
            scenario_id: Unique identifier for this scenario
            jammer_position: Position of the jammer
            target_position: Position of the target
            jammer_type: Type of jammer
            jammer_power: Jammer power in Watts ERP
            jammer_frequency: Jammer center frequency in Hz
            jammer_bandwidth: Jammer bandwidth in Hz
            target_frequency: Target receiver frequency in Hz
            target_bandwidth: Target receiver bandwidth in Hz
            target_sensitivity: Target receiver sensitivity in dBm
            terrain_loss: Additional loss due to terrain in dB
            atmospheric_loss: Additional loss due to atmosphere in dB
            
        Returns:
            The created jamming scenario
        """
        scenario = JammingScenario(
            jammer_position=jammer_position,
            target_position=target_position,
            jammer_type=jammer_type,
            jammer_power=jammer_power,
            jammer_frequency=jammer_frequency,
            jammer_bandwidth=jammer_bandwidth,
            target_frequency=target_frequency,
            target_bandwidth=target_bandwidth,
            target_sensitivity=target_sensitivity,
            terrain_loss=terrain_loss,
            atmospheric_loss=atmospheric_loss
        )
        
        self.scenarios[scenario_id] = scenario
        logger.info(f"Created jamming scenario {scenario_id}: {jammer_type.name} jammer at {jammer_frequency/1e6:.1f} MHz")
        return scenario
    
    def predict_effectiveness(self, scenario_id: str) -> JammingEffectiveness:
        """Predict jamming effectiveness for a given scenario.
        
        Args:
            scenario_id: ID of the scenario to analyze
            
        Returns:
            Jamming effectiveness metrics
        """
        if scenario_id not in self.scenarios:
            logger.error(f"Scenario {scenario_id} not found")
            raise ValueError(f"Scenario {scenario_id} not found")
            
        scenario = self.scenarios[scenario_id]
        
        # Calculate distance between jammer and target
        distance = scenario.jammer_position.distance_to(scenario.target_position)
        
        # Calculate free space path loss
        # FSPL (dB) = 20*log10(d) + 20*log10(f) - 147.55
        # where d is in meters and f is in Hz
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(scenario.jammer_frequency) - 147.55
        
        # Add terrain and atmospheric losses
        total_loss_db = path_loss_db + scenario.terrain_loss + scenario.atmospheric_loss
        
        # Calculate jammer power at target (dBm)
        jammer_power_dbm = 10 * np.log10(scenario.jammer_power * 1000)  # Convert W to dBm
        received_jammer_power_dbm = jammer_power_dbm - total_loss_db
        
        # Calculate frequency overlap factor (0-1)
        freq_diff = abs(scenario.jammer_frequency - scenario.target_frequency)
        combined_bandwidth = (scenario.jammer_bandwidth + scenario.target_bandwidth) / 2
        
        if freq_diff > combined_bandwidth:
            # No frequency overlap
            frequency_coverage = 0.0
        else:
            # Partial or complete overlap
            frequency_coverage = 1.0 - (freq_diff / combined_bandwidth)
            frequency_coverage = max(0.0, min(1.0, frequency_coverage))
        
        # Calculate effective jamming power based on bandwidth ratio
        if scenario.jammer_bandwidth > scenario.target_bandwidth:
            # Jammer power is spread over wider bandwidth
            bandwidth_ratio = scenario.target_bandwidth / scenario.jammer_bandwidth
            effective_jammer_power_dbm = received_jammer_power_dbm + 10 * np.log10(bandwidth_ratio)
        else:
            # All jammer power is within target bandwidth
            effective_jammer_power_dbm = received_jammer_power_dbm
            
        # Apply frequency coverage factor
        effective_jammer_power_dbm += 10 * np.log10(frequency_coverage)
        
        # Calculate J/S ratio (assuming target signal at sensitivity level)
        js_ratio_db = effective_jammer_power_dbm - scenario.target_sensitivity
        
        # Calculate probability of denial based on J/S ratio
        # Simplified model: sigmoid function centered at J/S = 10 dB
        probability_of_denial = 1.0 / (1.0 + np.exp(-(js_ratio_db - 10) / 3))
        
        # Calculate burn-through range
        # Range at which target signal overcomes jamming
        # Simplified calculation assuming 1/R^4 for monostatic radar
        if scenario.jammer_type in [JammerType.STANDOFF, JammerType.ESCORT]:
            # For standoff jamming, use 1/R^2 for jammer
            burn_through_range = distance * (10 ** (js_ratio_db / 20))
        else:
            # For self-protection or close-in jamming
            burn_through_range = distance * (10 ** (js_ratio_db / 40))
            
        # Calculate vulnerability score based on jammer type and target characteristics
        vulnerability_score = self._calculate_vulnerability_score(scenario)
        
        # Calculate countermeasure resistance
        countermeasure_resistance = self._calculate_countermeasure_resistance(scenario)
        
        # Calculate overall effectiveness
        overall_effectiveness = (
            0.3 * probability_of_denial +
            0.2 * frequency_coverage +
            0.2 * vulnerability_score +
            0.3 * countermeasure_resistance
        )
        
        # Create and store results
        results = JammingEffectiveness(
            jamming_to_signal_ratio=js_ratio_db,
            probability_of_denial=probability_of_denial,
            effective_range=distance,
            burn_through_range=burn_through_range,
            frequency_coverage=frequency_coverage,
            vulnerability_score=vulnerability_score,
            countermeasure_resistance=countermeasure_resistance,
            overall_effectiveness=overall_effectiveness
        )
        
        self.results[scenario_id] = results
        
        logger.info(f"Jamming scenario {scenario_id} analysis: J/S={js_ratio_db:.1f}dB, "
                   f"effectiveness={overall_effectiveness:.2f}")
        
        return results
    
    def _calculate_vulnerability_score(self, scenario: JammingScenario) -> float:
        """Calculate vulnerability score based on scenario parameters."""
        # Base vulnerability depends on jammer type
        if scenario.jammer_type == JammerType.BARRAGE:
            base_score = 0.7  # Good against many systems but power spread out
        elif scenario.jammer_type == JammerType.SPOT:
            base_score = 0.9  # Very effective when frequency is known
        elif scenario.jammer_type == JammerType.SWEEP:
            base_score = 0.6  # Moderate effectiveness
        elif scenario.jammer_type == JammerType.DECEPTIVE:
            base_score = 0.8  # Highly effective against tracking systems
        elif scenario.jammer_type == JammerType.REACTIVE:
            base_score = 0.75  # Good when properly triggered
        else:
            base_score = 0.5  # Default
            
        # Adjust based on frequency match
        freq_match_factor = 1.0 - 0.5 * abs(scenario.jammer_frequency - scenario.target_frequency) / scenario.target_frequency
        freq_match_factor = max(0.1, min(1.0, freq_match_factor))
        
        return base_score * freq_match_factor
    
    def _calculate_countermeasure_resistance(self, scenario: JammingScenario) -> float:
        """Calculate resistance to countermeasures."""
        # Base resistance depends on jammer type
        if scenario.jammer_type == JammerType.BARRAGE:
            base_resistance = 0.8  # Hard to frequency hop away from
        elif scenario.jammer_type == JammerType.SPOT:
            base_resistance = 0.4  # Vulnerable to frequency agility
        elif scenario.jammer_type == JammerType.SWEEP:
            base_resistance = 0.6  # Moderate resistance
        elif scenario.jammer_type == JammerType.DECEPTIVE:
            base_resistance = 0.7  # Fairly resistant
        elif scenario.jammer_type == JammerType.REACTIVE:
            base_resistance = 0.5  # Depends on reaction time
        else:
            base_resistance = 0.5  # Default
            
        # Adjust based on bandwidth
        bandwidth_factor = min(1.0, scenario.jammer_bandwidth / (5 * scenario.target_bandwidth))
        
        return base_resistance * (0.5 + 0.5 * bandwidth_factor)
    
    def create_ecm_parameters(self, scenario_id: str) -> ECMParameters:
        """Create ECM parameters from a jamming scenario for use with existing ECM systems.
        
        Args:
            scenario_id: ID of the scenario
            
        Returns:
            ECM parameters for the scenario
        """
        if scenario_id not in self.scenarios:
            logger.error(f"Scenario {scenario_id} not found")
            raise ValueError(f"Scenario {scenario_id} not found")
            
        scenario = self.scenarios[scenario_id]
        
        # Map jammer type to ECM technique
        if scenario.jammer_type in [JammerType.BARRAGE, JammerType.SPOT, JammerType.SWEEP]:
            technique = ECMTechniqueType.NOISE_JAMMING
        elif scenario.jammer_type == JammerType.DECEPTIVE:
            technique = ECMTechniqueType.DECEPTION_JAMMING
        else:
            technique = ECMTechniqueType.NOISE_JAMMING  # Default
            
        # Create ECM parameters
        ecm_params = ECMParameters(
            technique_type=technique,
            power_erp=scenario.jammer_power,
            bandwidth=scenario.jammer_bandwidth,
            duty_cycle=1.0,  # Assume continuous jamming
            parameters={
                "center_frequency": scenario.jammer_frequency,
                "jammer_type": scenario.jammer_type.name,
                "predicted_effectiveness": self.results.get(scenario_id, None)
            }
        )
        
        return ecm_params