"""Battlefield spectrum analysis and management tools."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
import matplotlib.pyplot as plt

from helios.utils.logger import get_logger
from helios.cognitive.spectrum_sensing import SpectrumSensor
from helios.dsp.spectral import power_spectrum
from helios.design.pulse_amplifier_designer import PulseAmplifierClass

logger = get_logger(__name__)

class SpectrumThreatLevel(Enum):
    """Threat levels for spectrum interference."""
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class BandStatus(Enum):
    """Status of frequency bands in battlefield environment."""
    CLEAR = auto()         # No significant activity
    FRIENDLY = auto()      # Friendly forces using band
    HOSTILE = auto()       # Hostile jamming or communications
    CONTESTED = auto()     # Both friendly and hostile activity
    RESERVED = auto()      # Reserved for specific operations
    UNUSABLE = auto()      # Too much interference or EMP effects


@dataclass
class FrequencyBand:
    """Frequency band with operational status."""
    min_freq: float        # Hz
    max_freq: float        # Hz
    status: BandStatus = BandStatus.CLEAR
    signal_power: float = -100.0  # dBm
    noise_floor: float = -120.0   # dBm
    threat_level: SpectrumThreatLevel = SpectrumThreatLevel.NONE
    priority: int = 0      # Priority for friendly use (higher = more important)


class BattlefieldSpectrumAnalyzer:
    """Analyzer for battlefield spectrum conditions."""
    
    def __init__(self):
        """Initialize the battlefield spectrum analyzer."""
        self.bands: List[FrequencyBand] = []
        self.spectrum_sensor = SpectrumSensor()
        self.threat_map: Dict[Tuple[float, float], SpectrumThreatLevel] = {}
        
    def define_band(self, min_freq: float, max_freq: float, 
                   status: BandStatus = BandStatus.CLEAR,
                   priority: int = 0) -> FrequencyBand:
        """Define a frequency band for monitoring.
        
        Args:
            min_freq: Minimum frequency in Hz
            max_freq: Maximum frequency in Hz
            status: Initial band status
            priority: Band priority (higher = more important)
            
        Returns:
            The created frequency band
        """
        band = FrequencyBand(
            min_freq=min_freq,
            max_freq=max_freq,
            status=status,
            priority=priority
        )
        self.bands.append(band)
        logger.info(f"Defined band {min_freq/1e6:.1f}-{max_freq/1e6:.1f} MHz with status {status.name}")
        return band
    
    def analyze_samples(self, samples: np.ndarray, sampling_rate: float, center_freq: float):
        """Analyze spectrum samples and update band statuses.
        
        Args:
            samples: Complex time-domain samples
            sampling_rate: Sampling rate in Hz
            center_freq: Center frequency in Hz
        """
        # Get frequency and power data
        freqs, power = power_spectrum(samples, sampling_rate)
        
        # Shift frequencies to actual center frequency
        freqs = freqs + center_freq
        
        # Update each band based on measured spectrum
        for band in self.bands:
            # Find samples within this band
            mask = (freqs >= band.min_freq) & (freqs <= band.max_freq)
            if not np.any(mask):
                continue
                
            # Calculate band metrics
            band_power = power[mask]
            avg_power_db = 10 * np.log10(np.mean(band_power) + 1e-10)
            peak_power_db = 10 * np.log10(np.max(band_power) + 1e-10)
            
            # Update band information
            band.signal_power = peak_power_db
            
            # Determine threat level based on power and previous status
            self._update_threat_level(band, avg_power_db, peak_power_db)
            
        logger.info(f"Analyzed spectrum with {len(self.bands)} bands")
    
    def _update_threat_level(self, band: FrequencyBand, avg_power_db: float, peak_power_db: float):
        """Update threat level based on signal characteristics."""
        # Simple threshold-based classification
        snr = peak_power_db - band.noise_floor
        
        if snr < 3:
            band.threat_level = SpectrumThreatLevel.NONE
        elif snr < 10:
            band.threat_level = SpectrumThreatLevel.LOW
        elif snr < 20:
            band.threat_level = SpectrumThreatLevel.MEDIUM
        elif snr < 30:
            band.threat_level = SpectrumThreatLevel.HIGH
        else:
            band.threat_level = SpectrumThreatLevel.CRITICAL
            
        # Update band status based on threat and existing knowledge
        if band.status == BandStatus.FRIENDLY and band.threat_level >= SpectrumThreatLevel.HIGH:
            band.status = BandStatus.CONTESTED
            logger.warning(f"Band {band.min_freq/1e6:.1f}-{band.max_freq/1e6:.1f} MHz now contested")
    
    def recommend_frequencies(self, bandwidth: float, 
                             application: PulseAmplifierClass) -> List[Tuple[float, float]]:
        """Recommend available frequencies for a specific application.
        
        Args:
            bandwidth: Required bandwidth in Hz
            application: Type of application
            
        Returns:
            List of (center_freq, bandwidth) tuples in Hz
        """
        recommendations = []
        
        # Filter bands by status
        available_bands = [b for b in self.bands if b.status in 
                          [BandStatus.CLEAR, BandStatus.FRIENDLY]]
        
        # Sort by threat level (ascending) and priority (descending)
        available_bands.sort(key=lambda b: (b.threat_level.value, -b.priority))
        
        for band in available_bands:
            band_width = band.max_freq - band.min_freq
            
            # Skip if band is too narrow
            if band_width < bandwidth:
                continue
                
            # For EW applications, prioritize bands with hostile activity
            if application == PulseAmplifierClass.ELECTRONIC_WARFARE and band.status != BandStatus.HOSTILE:
                continue
                
            # For radar, avoid contested bands
            if application == PulseAmplifierClass.RADAR and band.status == BandStatus.CONTESTED:
                continue
                
            # Calculate how many channels we can fit
            num_channels = int(band_width / bandwidth)
            channel_width = band_width / num_channels
            
            for i in range(num_channels):
                center = band.min_freq + (i + 0.5) * channel_width
                recommendations.append((center, bandwidth))
                
                # Limit to top 5 recommendations
                if len(recommendations) >= 5:
                    break
                    
            if len(recommendations) >= 5:
                break
                
        return recommendations
    
    def get_band_status_summary(self) -> Dict[BandStatus, int]:
        """Get summary of band statuses.
        
        Returns:
            Dictionary of status to count
        """
        summary = {status: 0 for status in BandStatus}
        for band in self.bands:
            summary[band.status] += 1
        return summary


class BattlefieldSpectrumManager:
    """Manager for battlefield spectrum allocation and coordination."""
    
    def __init__(self):
        """Initialize the battlefield spectrum manager."""
        self.analyzer = BattlefieldSpectrumAnalyzer()
        self.allocated_bands: Dict[str, FrequencyBand] = {}
        self.frequency_plan: Dict[str, List[Tuple[float, float]]] = {}
        
    def initialize_common_bands(self):
        """Initialize common military frequency bands."""
        # VHF Tactical
        self.analyzer.define_band(30e6, 88e6, BandStatus.RESERVED, priority=8)
        
        # UHF Tactical
        self.analyzer.define_band(225e6, 400e6, BandStatus.FRIENDLY, priority=9)
        
        # L-band Radar
        self.analyzer.define_band(1.0e9, 2.0e9, BandStatus.FRIENDLY, priority=7)
        
        # S-band Radar
        self.analyzer.define_band(2.0e9, 4.0e9, BandStatus.CLEAR, priority=6)
        
        # C-band Radar/Comms
        self.analyzer.define_band(4.0e9, 8.0e9, BandStatus.CLEAR, priority=5)
        
        # X-band Radar
        self.analyzer.define_band(8.0e9, 12.0e9, BandStatus.FRIENDLY, priority=8)
        
        # Ku-band Satellite
        self.analyzer.define_band(12.0e9, 18.0e9, BandStatus.RESERVED, priority=7)
        
        logger.info("Initialized common military frequency bands")
    
    def allocate_band(self, system_id: str, bandwidth: float, 
                     application: PulseAmplifierClass) -> Optional[Tuple[float, float]]:
        """Allocate a frequency band for a specific system.
        
        Args:
            system_id: System identifier
            bandwidth: Required bandwidth in Hz
            application: Type of application
            
        Returns:
            Tuple of (center_freq, bandwidth) if successful, None otherwise
        """
        recommendations = self.analyzer.recommend_frequencies(bandwidth, application)
        
        if not recommendations:
            logger.warning(f"No suitable frequency bands available for {system_id}")
            return None
            
        # Take the first recommendation
        center_freq, alloc_bandwidth = recommendations[0]
        
        # Find which band this belongs to
        for band in self.analyzer.bands:
            if band.min_freq <= center_freq <= band.max_freq:
                # Record the allocation
                self.allocated_bands[system_id] = band
                
                # If this was a clear band, mark it as friendly now
                if band.status == BandStatus.CLEAR:
                    band.status = BandStatus.FRIENDLY
                    
                logger.info(f"Allocated {center_freq/1e6:.2f} MHz with {alloc_bandwidth/1e6:.2f} MHz bandwidth to {system_id}")
                return center_freq, alloc_bandwidth
                
        return None
    
    def release_allocation(self, system_id: str):
        """Release a frequency allocation.
        
        Args:
            system_id: System identifier
        """
        if system_id in self.allocated_bands:
            band = self.allocated_bands[system_id]
            
            # Check if we need to update the band status
            # This is simplified - in reality would need to track all allocations per band
            logger.info(f"Released allocation for {system_id}")
            del self.allocated_bands[system_id]