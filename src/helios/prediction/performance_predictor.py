"""Performance prediction module for RF systems."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from helios.design.rf_system_design import RFSystemDesign
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for system performance metrics."""
    snr: float = 0.0
    throughput: float = 0.0
    bit_error_rate: float = 0.0
    latency: float = 0.0
    power_consumption: float = 0.0
    spectral_efficiency: float = 0.0
    range: float = 0.0

class PerformancePredictor:
    """Predicts RF system performance under various conditions."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the performance predictor.
        
        Args:
            design: RF system design to analyze
        """
        self.design = design
        
    def predict_performance(self, 
                           environment_conditions: Dict[str, Any],
                           operational_parameters: Dict[str, Any]) -> PerformanceMetrics:
        """Predict system performance under specified conditions.
        
        Args:
            environment_conditions: Dict of environmental factors (noise, interference, etc.)
            operational_parameters: Dict of operational settings (power, frequency, etc.)
            
        Returns:
            Predicted performance metrics
        """
        # Extract key parameters
        noise_floor_dbm = environment_conditions.get('noise_floor_dbm', -100)
        interference_dbm = environment_conditions.get('interference_dbm', -110)
        distance_m = environment_conditions.get('distance_m', 1000)
        frequency_hz = operational_parameters.get('frequency_hz', 1e9)
        tx_power_dbm = operational_parameters.get('tx_power_dbm', 30)
        bandwidth_hz = operational_parameters.get('bandwidth_hz', 1e6)
        modulation = operational_parameters.get('modulation', 'QPSK')
        
        # Calculate link budget
        path_loss = self._calculate_path_loss(distance_m, frequency_hz)
        rx_power_dbm = tx_power_dbm - path_loss
        
        # Calculate SNR
        total_noise_dbm = 10 * np.log10(10**(noise_floor_dbm/10) + 10**(interference_dbm/10))
        snr_db = rx_power_dbm - total_noise_dbm
        
        # Calculate other metrics
        bit_error_rate = self._estimate_ber(snr_db, modulation)
        throughput = self._estimate_throughput(bandwidth_hz, snr_db, modulation)
        spectral_efficiency = throughput / bandwidth_hz if bandwidth_hz > 0 else 0
        
        # Estimate power consumption (simplified)
        power_consumption = self._estimate_power_consumption(tx_power_dbm)
        
        # Estimate latency (simplified)
        latency = self._estimate_latency(distance_m, bit_error_rate)
        
        # Calculate maximum range
        max_range = self._calculate_max_range(tx_power_dbm, frequency_hz, 
                                            noise_floor_dbm, min_snr_db=10)
        
        return PerformanceMetrics(
            snr=snr_db,
            throughput=throughput,
            bit_error_rate=bit_error_rate,
            latency=latency,
            power_consumption=power_consumption,
            spectral_efficiency=spectral_efficiency,
            range=max_range
        )
    
    def _calculate_path_loss(self, distance_m: float, frequency_hz: float) -> float:
        """Calculate free space path loss.
        
        Args:
            distance_m: Distance in meters
            frequency_hz: Frequency in Hz
            
        Returns:
            Path loss in dB
        """
        # Free space path loss formula: 20*log10(d) + 20*log10(f) + 32.44
        # where d is in km and f is in MHz
        distance_km = distance_m / 1000
        frequency_mhz = frequency_hz / 1e6
        return 20 * np.log10(distance_km) + 20 * np.log10(frequency_mhz) + 32.44
    
    def _estimate_ber(self, snr_db: float, modulation: str) -> float:
        """Estimate bit error rate based on SNR and modulation.
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            modulation: Modulation scheme
            
        Returns:
            Estimated bit error rate
        """
        # Simple BER estimation (can be expanded with more accurate models)
        snr_linear = 10 ** (snr_db / 10)
        
        if modulation == 'BPSK':
            return 0.5 * np.exp(-snr_linear)
        elif modulation == 'QPSK':
            return 0.5 * np.exp(-snr_linear / 2)
        elif modulation == '16QAM':
            return 0.2 * np.exp(-snr_linear / 8)
        elif modulation == '64QAM':
            return 0.1 * np.exp(-snr_linear / 16)
        else:
            # Default case
            return np.exp(-snr_linear / 4)
    
    def _estimate_throughput(self, bandwidth_hz: float, snr_db: float, modulation: str) -> float:
        """Estimate throughput based on bandwidth, SNR and modulation.
        
        Args:
            bandwidth_hz: Bandwidth in Hz
            snr_db: Signal-to-noise ratio in dB
            modulation: Modulation scheme
            
        Returns:
            Estimated throughput in bits per second
        """
        # Shannon capacity as upper bound
        shannon_capacity = bandwidth_hz * np.log2(1 + 10 ** (snr_db / 10))
        
        # Modulation efficiency factors (bits per symbol)
        modulation_efficiency = {
            'BPSK': 1,
            'QPSK': 2,
            '8PSK': 3,
            '16QAM': 4,
            '64QAM': 6,
            '256QAM': 8
        }
        
        # Get bits per symbol for the modulation
        bits_per_symbol = modulation_efficiency.get(modulation, 2)
        
        # Practical throughput with coding overhead
        coding_rate = 0.75  # Typical coding rate
        practical_throughput = bandwidth_hz * bits_per_symbol * coding_rate
        
        # Return the minimum of Shannon capacity and practical throughput
        return min(shannon_capacity, practical_throughput)
    
    def _estimate_power_consumption(self, tx_power_dbm: float) -> float:
        """Estimate system power consumption.
        
        Args:
            tx_power_dbm: Transmit power in dBm
            
        Returns:
            Estimated power consumption in Watts
        """
        # Convert dBm to Watts
        tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10)
        
        # Simplified power amplifier efficiency model
        pa_efficiency = 0.35  # 35% efficiency
        pa_power = tx_power_watts / pa_efficiency
        
        # Add base power consumption for other components
        base_power = 2.0  # 2W for digital processing, etc.
        
        return pa_power + base_power
    
    def _estimate_latency(self, distance_m: float, ber: float) -> float:
        """Estimate system latency.
        
        Args:
            distance_m: Distance in meters
            ber: Bit error rate
            
        Returns:
            Estimated latency in seconds
        """
        # Propagation delay (speed of light)
        speed_of_light = 3e8  # m/s
        propagation_delay = distance_m / speed_of_light
        
        # Processing delay (simplified)
        processing_delay = 0.001  # 1ms
        
        # Retransmission delay due to errors
        retransmission_factor = 1 / (1 - min(ber * 1000, 0.9))  # Cap at 0.9 to avoid infinity
        
        return propagation_delay + processing_delay * retransmission_factor
    
    def _calculate_max_range(self, tx_power_dbm: float, frequency_hz: float, 
                           noise_floor_dbm: float, min_snr_db: float = 10) -> float:
        """Calculate maximum communication range.
        
        Args:
            tx_power_dbm: Transmit power in dBm
            frequency_hz: Frequency in Hz
            noise_floor_dbm: Noise floor in dBm
            min_snr_db: Minimum required SNR in dB
            
        Returns:
            Maximum range in meters
        """
        # Maximum allowed path loss
        max_path_loss = tx_power_dbm - noise_floor_dbm - min_snr_db
        
        # Rearrange free space path loss formula to solve for distance
        frequency_mhz = frequency_hz / 1e6
        term = max_path_loss - 20 * np.log10(frequency_mhz) - 32.44
        distance_km = 10 ** (term / 20)
        
        return distance_km * 1000  # Convert to meters