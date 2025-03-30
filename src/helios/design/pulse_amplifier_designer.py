"""Power amplifier design tools optimized for pulse operation."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from helios.design.rf_components import Amplifier, RFComponent
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class PulseAmplifierClass(Enum):
    """Classification of pulse amplifiers by application."""
    RADAR = auto()        # Radar pulse amplifiers
    ELECTRONIC_WARFARE = auto()  # EW applications
    DIRECTED_ENERGY = auto()     # Directed energy weapons
    COMMUNICATIONS = auto()      # Pulsed communications
    TEST_EQUIPMENT = auto()      # Lab/test equipment


@dataclass
class PulseParameters:
    """Parameters specific to pulsed operation."""
    pulse_width: float = 1e-6      # seconds
    duty_cycle: float = 0.1        # ratio (0-1)
    rise_time: float = 100e-9      # seconds
    fall_time: float = 100e-9      # seconds
    droop: float = 0.1             # amplitude droop during pulse (ratio)
    recovery_time: float = 1e-6    # seconds to recover between pulses


class PulseAmplifierDesigner:
    """Designer for power amplifiers optimized for pulse operation."""
    
    def __init__(self):
        """Initialize the pulse amplifier designer."""
        self.designs: Dict[str, Dict[str, Any]] = {}
        
    def design_pulse_amplifier(self,
                              name: str,
                              frequency: float,
                              peak_power: float,
                              pulse_params: PulseParameters,
                              amp_class: PulseAmplifierClass = PulseAmplifierClass.RADAR) -> Amplifier:
        """
        Design a power amplifier optimized for pulse operation.
        
        Args:
            name: Amplifier name
            frequency: Operating frequency in Hz
            peak_power: Peak output power in Watts
            pulse_params: Pulse operation parameters
            amp_class: Amplifier classification
            
        Returns:
            Configured amplifier component
        """
        # Create base amplifier
        amplifier = Amplifier(name=name)
        
        # Calculate peak power in dBm
        peak_power_dbm = 10 * np.log10(peak_power * 1000)
        
        # Set basic parameters
        amplifier.gain = self._calculate_gain(peak_power)
        amplifier.p1db = peak_power_dbm - 2  # Typical P1dB is slightly below peak power
        amplifier.noise_figure = self._estimate_noise_figure(frequency, peak_power)
        amplifier.frequency_range = (frequency * 0.9, frequency * 1.1)  # ±10% bandwidth
        
        # Calculate thermal parameters based on duty cycle
        average_power = peak_power * pulse_params.duty_cycle
        thermal_dissipation = average_power * 0.5  # Assuming 50% efficiency
        
        # Store design details
        self.designs[amplifier.id] = {
            "peak_power_w": peak_power,
            "peak_power_dbm": peak_power_dbm,
            "average_power_w": average_power,
            "thermal_dissipation_w": thermal_dissipation,
            "pulse_params": pulse_params,
            "amplifier_class": amp_class,
            "technology": self._select_technology(frequency, peak_power),
            "efficiency": self._estimate_efficiency(pulse_params.duty_cycle, peak_power)
        }
        
        logger.info(f"Designed pulse amplifier: {name}, {peak_power:.1f}W peak, "
                   f"{pulse_params.duty_cycle*100:.1f}% duty cycle")
        return amplifier
    
    def get_design_details(self, amplifier_id: str) -> Dict[str, Any]:
        """Get detailed design information for an amplifier."""
        if amplifier_id not in self.designs:
            logger.warning(f"No design details found for amplifier {amplifier_id}")
            return {}
        return self.designs[amplifier_id]
    
    def _calculate_gain(self, peak_power: float) -> float:
        """Calculate appropriate gain based on peak power."""
        # Simple heuristic: higher power typically means lower gain
        if peak_power > 1000:  # >1kW
            return 10.0
        elif peak_power > 100:  # >100W
            return 15.0
        elif peak_power > 10:   # >10W
            return 20.0
        else:
            return 25.0
    
    def _estimate_noise_figure(self, frequency: float, peak_power: float) -> float:
        """Estimate noise figure based on frequency and power level."""
        # Simple heuristic based on frequency band and power
        base_nf = 3.0
        
        # Higher frequencies typically have higher noise figures
        if frequency > 10e9:  # X-band and above
            base_nf += 2.0
        elif frequency > 1e9:  # L/S/C-band
            base_nf += 1.0
            
        # Higher power amps typically have higher noise figures
        if peak_power > 100:
            base_nf += 1.0
            
        return base_nf
    
    def _select_technology(self, frequency: float, peak_power: float) -> str:
        """Select appropriate semiconductor technology based on frequency and power."""
        if frequency > 20e9:  # >20 GHz
            return "GaN HEMT" if peak_power > 10 else "GaAs pHEMT"
        elif frequency > 5e9:  # >5 GHz
            if peak_power > 100:
                return "GaN HEMT"
            elif peak_power > 10:
                return "GaN/GaAs"
            else:
                return "GaAs"
        else:  # <5 GHz
            if peak_power > 1000:
                return "LDMOS/GaN"
            elif peak_power > 100:
                return "LDMOS"
            else:
                return "GaN/Si"
    
    def _estimate_efficiency(self, duty_cycle: float, peak_power: float) -> float:
        """Estimate amplifier efficiency based on duty cycle and power level."""
        # Base efficiency - higher power typically means higher efficiency
        if peak_power > 1000:
            base_efficiency = 0.65  # 65%
        elif peak_power > 100:
            base_efficiency = 0.60  # 60%
        elif peak_power > 10:
            base_efficiency = 0.55  # 55%
        else:
            base_efficiency = 0.50  # 50%
            
        # Adjust for duty cycle - very low duty cycles can have higher peak efficiency
        if duty_cycle < 0.01:  # <1%
            efficiency_factor = 1.1  # 10% boost
        elif duty_cycle < 0.1:  # <10%
            efficiency_factor = 1.05  # 5% boost
        else:
            efficiency_factor = 1.0  # no adjustment
            
        return base_efficiency * efficiency_factor