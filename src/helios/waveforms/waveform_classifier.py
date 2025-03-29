"""Waveform classification and identification module."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum, auto
from scipy import signal
import logging

from helios.core.data_structures import Waveform, ModulationType
from helios.waveforms.tactical_waveforms import (
    Link16Waveform, SINCGARSWaveform, HaveQuickWaveform, MILSTDWaveform
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class WaveformClass(Enum):
    """Classification categories for waveforms."""
    UNKNOWN = auto()
    CONTINUOUS_WAVE = auto()
    FREQUENCY_HOPPING = auto()
    DIRECT_SEQUENCE_SPREAD_SPECTRUM = auto()
    OFDM = auto()
    PULSED = auto()
    AM = auto()
    FM = auto()
    PSK = auto()
    QAM = auto()
    TACTICAL_LINK16 = auto()
    TACTICAL_SINCGARS = auto()
    TACTICAL_HAVEQUICK = auto()
    TACTICAL_MILSTD = auto()

class WaveformClassifier:
    """Identifies and classifies waveforms from signal samples."""
    
    def __init__(self, sampling_rate: float = 1e6):
        """Initialize the waveform classifier.
        
        Args:
            sampling_rate: Default sampling rate for analysis in Hz
        """
        self.sampling_rate = sampling_rate
        
    def classify(self, samples: np.ndarray, sampling_rate: Optional[float] = None) -> Tuple[WaveformClass, Dict[str, Any]]:
        """
        Classify a waveform from time-domain samples.
        
        Args:
            samples: Complex time-domain samples
            sampling_rate: Sample rate in Hz (uses default if None)
        
        Returns:
            Tuple of (waveform class, parameters dict)
        """
        fs = sampling_rate or self.sampling_rate
        logger.info(f"Classifying waveform from {len(samples)} samples at {fs/1e6} MHz")
        
        # Extract basic features
        features = self._extract_features(samples, fs)
        
        # Detect frequency hopping
        if self._is_frequency_hopping(samples, fs):
            # Further classify the specific type of frequency hopping waveform
            if self._matches_link16_pattern(samples, fs):
                return WaveformClass.TACTICAL_LINK16, self._estimate_link16_params(samples, fs)
            elif self._matches_sincgars_pattern(samples, fs):
                return WaveformClass.TACTICAL_SINCGARS, self._estimate_sincgars_params(samples, fs)
            elif self._matches_havequick_pattern(samples, fs):
                return WaveformClass.TACTICAL_HAVEQUICK, self._estimate_havequick_params(samples, fs)
            else:
                return WaveformClass.FREQUENCY_HOPPING, self._estimate_fh_params(samples, fs)
        
        # Detect modulation type
        mod_type = self._detect_modulation(samples, fs)
        if mod_type == ModulationType.AM:
            return WaveformClass.AM, self._estimate_am_params(samples, fs)
        elif mod_type == ModulationType.FM:
            return WaveformClass.FM, self._estimate_fm_params(samples, fs)
        elif mod_type == ModulationType.PSK:
            return WaveformClass.PSK, self._estimate_psk_params(samples, fs)
        
        # Check for OFDM characteristics
        if self._is_ofdm(samples, fs):
            return WaveformClass.OFDM, self._estimate_ofdm_params(samples, fs)
        
        # Check for spread spectrum
        if self._is_spread_spectrum(samples, fs):
            return WaveformClass.DIRECT_SEQUENCE_SPREAD_SPECTRUM, self._estimate_dsss_params(samples, fs)
        
        # Check for MIL-STD waveforms
        if self._matches_milstd_pattern(samples, fs):
            return WaveformClass.TACTICAL_MILSTD, self._estimate_milstd_params(samples, fs)
        
        # Default to continuous wave if nothing else matches
        if np.var(np.abs(samples)) < 0.01:  # Low amplitude variance
            return WaveformClass.CONTINUOUS_WAVE, features
            
        return WaveformClass.UNKNOWN, features
    
    def identify_waveform(self, samples: np.ndarray, sampling_rate: Optional[float] = None) -> Optional[Waveform]:
        """
        Identify a specific waveform type from samples and create a Waveform object.
        
        Args:
            samples: Complex time-domain samples
            sampling_rate: Sample rate in Hz (uses default if None)
            
        Returns:
            A Waveform object of the appropriate subclass, or None if unidentified
        """
        waveform_class, params = self.classify(samples, sampling_rate)
        fs = sampling_rate or self.sampling_rate
        
        # Create appropriate waveform object based on classification
        if waveform_class == WaveformClass.TACTICAL_LINK16:
            return Link16Waveform(
                center_frequency=params.get('center_frequency', 0.0),
                bandwidth=params.get('bandwidth', 3e6),
                amplitude=params.get('amplitude', 1.0),
                pulse_repetition_frequency=params.get('prf', 50e3),
                frequency_hopping_pattern=params.get('hop_pattern', 'standard'),
                transmission_security_level=params.get('security_level', 1)
            )
        elif waveform_class == WaveformClass.TACTICAL_SINCGARS:
            return SINCGARSWaveform(
                center_frequency=params.get('center_frequency', 0.0),
                bandwidth=params.get('bandwidth', 25e3),
                amplitude=params.get('amplitude', 1.0),
                hop_rate=params.get('hop_rate', 100.0),
                mode=params.get('mode', 'frequency_hopping')
            )
        elif waveform_class == WaveformClass.TACTICAL_HAVEQUICK:
            return HaveQuickWaveform(
                center_frequency=params.get('center_frequency', 0.0),
                bandwidth=params.get('bandwidth', 25e3),
                amplitude=params.get('amplitude', 1.0),
                dwell_time=params.get('dwell_time', 0.01)
            )
        elif waveform_class == WaveformClass.TACTICAL_MILSTD:
            return MILSTDWaveform(
                center_frequency=params.get('center_frequency', 0.0),
                bandwidth=params.get('bandwidth', 3e3),
                amplitude=params.get('amplitude', 1.0),
                standard_type=params.get('standard_type', '188-110A'),
                data_rate=params.get('data_rate', 1200.0)
            )
        elif waveform_class != WaveformClass.UNKNOWN:
            # Create a basic waveform for other recognized types
            return Waveform(
                center_frequency=params.get('center_frequency', 0.0),
                bandwidth=params.get('bandwidth', 0.0),
                amplitude=params.get('amplitude', 1.0),
                modulation_type=self._class_to_modulation_type(waveform_class),
                modulation_params=params
            )
        
        return None
    
    def _extract_features(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Extract basic features from signal samples."""
        # Calculate basic time domain features
        amplitude = np.mean(np.abs(samples))
        
        # Calculate frequency domain features
        f, psd = signal.welch(samples, fs, nperseg=min(1024, len(samples)))
        center_idx = np.argmax(psd)
        center_frequency = f[center_idx]
        
        # Estimate bandwidth (simple method - can be improved)
        psd_db = 10 * np.log10(psd / np.max(psd) + 1e-10)
        mask = psd_db > -20  # -20 dB threshold
        if np.sum(mask) > 1:
            bandwidth = f[mask][-1] - f[mask][0]
        else:
            bandwidth = 0.0
            
        return {
            'amplitude': amplitude,
            'center_frequency': center_frequency,
            'bandwidth': bandwidth,
            'psd': psd,
            'frequencies': f
        }
    
    def _is_frequency_hopping(self, samples: np.ndarray, fs: float) -> bool:
        """Detect if the signal is using frequency hopping."""
        # Simple detection using spectrogram
        f, t, Sxx = signal.spectrogram(samples, fs, nperseg=min(256, len(samples)//10))
        
        # Look for distinct frequency changes
        peak_freqs = np.argmax(Sxx, axis=0)
        freq_changes = np.diff(peak_freqs)
        
        # If we see significant jumps in frequency, it's likely frequency hopping
        return np.max(np.abs(freq_changes)) > 5
    
    def _detect_modulation(self, samples: np.ndarray, fs: float) -> ModulationType:
        """Detect the modulation type of the signal."""
        # Extract amplitude and phase
        amplitude = np.abs(samples)
        phase = np.unwrap(np.angle(samples))
        
        # Check amplitude variation for AM
        amp_var = np.var(amplitude) / np.mean(amplitude)**2
        if amp_var > 0.05:
            return ModulationType.AM
            
        # Check phase variation for FM/PM
        phase_diff = np.diff(phase)
        if np.var(phase_diff) > 0.1:
            return ModulationType.FM
            
        # Check for PSK (simplified)
        # Look for distinct phase clusters
        phase_hist, _ = np.histogram(phase % (2*np.pi), bins=36)
        peaks = signal.find_peaks(phase_hist, height=len(samples)/100)[0]
        if len(peaks) >= 2 and len(peaks) <= 16:
            return ModulationType.PSK
            
        return ModulationType.NONE
    
    def _is_ofdm(self, samples: np.ndarray, fs: float) -> bool:
        """Detect if the signal is using OFDM modulation."""
        # Look for cyclic prefix correlation
        # This is a simplified approach - real OFDM detection is more complex
        for cp_len in [16, 32, 64, 128, 256]:
            if len(samples) < 3*cp_len:
                continue
                
            # Check correlation between potential CP and corresponding symbol part
            corr = []
            for i in range(0, len(samples)-2*cp_len, cp_len):
                c = np.abs(np.corrcoef(
                    np.abs(samples[i:i+cp_len]), 
                    np.abs(samples[i+cp_len:i+2*cp_len])
                )[0,1])
                corr.append(c)
                
            if len(corr) > 2 and np.mean(corr) > 0.7:
                return True
                
        return False
    
    def _is_spread_spectrum(self, samples: np.ndarray, fs: float) -> bool:
        """Detect if the signal is using spread spectrum techniques."""
        # Check for wideband characteristics with low power spectral density
        f, psd = signal.welch(samples, fs, nperseg=min(1024, len(samples)))
        
        # Spread spectrum signals typically have wide bandwidth and relatively flat PSD
        psd_db = 10 * np.log10(psd / np.max(psd) + 1e-10)
        mask = psd_db > -10  # -10 dB threshold
        
        # Check if bandwidth is wide and PSD is relatively flat
        if np.sum(mask) > len(psd) / 4:  # Using at least 25% of spectrum
            psd_std = np.std(psd[mask])
            psd_mean = np.mean(psd[mask])
            if psd_std / psd_mean < 0.5:  # Relatively flat PSD
                return True
                
        return False
    
    # Simplified implementations of the remaining methods
    def _matches_link16_pattern(self, samples: np.ndarray, fs: float) -> bool:
        """Check if the signal matches Link-16 characteristics."""
        # Simplified implementation - would need more sophisticated analysis in practice
        return self._is_frequency_hopping(samples, fs) and self._estimate_hop_interval(samples, fs) < 0.002
    
    def _matches_sincgars_pattern(self, samples: np.ndarray, fs: float) -> bool:
        """Check if the signal matches SINCGARS characteristics."""
        hop_interval = self._estimate_hop_interval(samples, fs)
        return self._is_frequency_hopping(samples, fs) and 0.005 < hop_interval < 0.02
    
    def _matches_havequick_pattern(self, samples: np.ndarray, fs: float) -> bool:
        """Check if the signal matches HAVE QUICK characteristics."""
        hop_interval = self._estimate_hop_interval(samples, fs)
        return self._is_frequency_hopping(samples, fs) and 0.005 < hop_interval < 0.05
    
    def _matches_milstd_pattern(self, samples: np.ndarray, fs: float) -> bool:
        """Check if the signal matches MIL-STD characteristics."""
        # Simplified check for MIL-STD waveforms
        mod_type = self._detect_modulation(samples, fs)
        features = self._extract_features(samples, fs)
        return mod_type == ModulationType.PSK and features['bandwidth'] < 10e3
    
    def _estimate_hop_interval(self, samples: np.ndarray, fs: float) -> float:
        """Estimate the hop interval for frequency hopping signals."""
        # Simple estimation using spectrogram
        f, t, Sxx = signal.spectrogram(samples, fs, nperseg=min(256, len(samples)//10))
        
        # Find frequency transitions
        peak_freqs = np.argmax(Sxx, axis=0)
        transitions = np.where(np.abs(np.diff(peak_freqs)) > 3)[0]
        
        if len(transitions) < 2:
            return 0.0
            
        # Calculate average time between transitions
        hop_times = np.diff(transitions) * (t[1] - t[0])
        return np.mean(hop_times)
    
    # Simplified parameter estimation methods
    def _estimate_link16_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate Link-16 parameters."""
        features = self._extract_features(samples, fs)
        features['hop_interval'] = self._estimate_hop_interval(samples, fs)
        features['prf'] = self._estimate_pulse_repetition_frequency(samples, fs)
        features['hop_pattern'] = 'standard'
        return features
    
    def _estimate_sincgars_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate SINCGARS parameters."""
        features = self._extract_features(samples, fs)
        hop_interval = self._estimate_hop_interval(samples, fs)
        features['hop_rate'] = 1.0 / hop_interval if hop_interval > 0 else 0
        features['mode'] = 'frequency_hopping'
        return features
    
    def _estimate_havequick_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate HAVE QUICK parameters."""
        features = self._extract_features(samples, fs)
        features['dwell_time'] = self._estimate_hop_interval(samples, fs)
        return features
    
    def _estimate_milstd_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate MIL-STD parameters."""
        features = self._extract_features(samples, fs)
        # Estimate data rate from symbol transitions
        features['data_rate'] = self._estimate_symbol_rate(samples, fs)
        features['standard_type'] = '188-110A'  # Default
        return features
    
    def _estimate_fh_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate generic frequency hopping parameters."""
        features = self._extract_features(samples, fs)
        features['hop_interval'] = self._estimate_hop_interval(samples, fs)
        return features
    
    def _estimate_am_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate AM modulation parameters."""
        features = self._extract_features(samples, fs)
        amplitude = np.abs(samples)
        
        # Estimate modulation frequency
        f_amp, psd_amp = signal.welch(amplitude, fs, nperseg=min(1024, len(samples)))
        mod_idx = np.argmax(psd_amp[1:]) + 1  # Skip DC component
        features['modulation_frequency'] = f_amp[mod_idx]
        
        # Estimate modulation index
        env_max = np.max(amplitude)
        env_min = np.min(amplitude)
        features['modulation_index'] = (env_max - env_min) / (env_max + env_min)
        
        return features
    
    def _estimate_fm_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate FM modulation parameters."""
        features = self._extract_features(samples, fs)
        
        # Extract instantaneous frequency
        phase = np.unwrap(np.angle(samples))
        inst_freq = np.diff(phase) * fs / (2 * np.pi)
        
        # Estimate modulation frequency
        f_freq, psd_freq = signal.welch(inst_freq, fs, nperseg=min(1024, len(inst_freq)))
        mod_idx = np.argmax(psd_freq[1:]) + 1  # Skip DC component
        features['modulation_frequency'] = f_freq[mod_idx]
        
        # Estimate frequency deviation
        features['frequency_deviation'] = np.std(inst_freq)
        
        return features
    
    def _estimate_psk_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate PSK modulation parameters."""
        features = self._extract_features(samples, fs)
        
        # Estimate number of phase states
        phase = np.angle(samples) % (2 * np.pi)
        phase_hist, _ = np.histogram(phase, bins=36)
        peaks = signal.find_peaks(phase_hist, height=len(samples)/100)[0]
        features['num_phases'] = len(peaks)
        
        # Estimate symbol rate
        features['symbol_rate'] = self._estimate_symbol_rate(samples, fs)
        
        return features
    
    def _estimate_ofdm_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate OFDM parameters."""
        features = self._extract_features(samples, fs)
        
        # Simplified OFDM parameter estimation
        # In practice, this would be much more complex
        features['num_subcarriers'] = 64  # Default estimate
        features['cyclic_prefix_length'] = 16  # Default estimate
        
        return features
    
    def _estimate_dsss_params(self, samples: np.ndarray, fs: float) -> Dict[str, Any]:
        """Estimate Direct Sequence Spread Spectrum parameters."""
        features = self._extract_features(samples, fs)
        
        # Simplified DSSS parameter estimation
        features['chip_rate'] = fs / 10  # Very rough estimate
        features['processing_gain'] = 10.0  # Default estimate
        
        return features
    
    def _estimate_symbol_rate(self, samples: np.ndarray, fs: float) -> float:
        """Estimate symbol rate from signal transitions."""
        # Simple method using amplitude changes
        amplitude = np.abs(samples)
        # Normalize and threshold
        norm_amp = (amplitude - np.min(amplitude)) / (np.max(amplitude) - np.min(amplitude))
        # Find transitions
        transitions = np.where(np.abs(np.diff(norm_amp)) > 0.2)[0]
        
        if len(transitions) < 2:
            return 1200.0  # Default value
            
        # Calculate average time between transitions
        transition_intervals = np.diff(transitions)
        avg_samples_per_symbol = np.mean(transition_intervals)
        
        return float(fs / avg_samples_per_symbol)
    
    def _estimate_pulse_repetition_frequency(self, samples: np.ndarray, fs: float) -> float:
        """Estimate pulse repetition frequency."""
        # Simple method using amplitude envelope
        amplitude = np.abs(samples)
        # Find pulses using threshold
        threshold = np.mean(amplitude) + 0.5 * np.std(amplitude)
        pulse_mask = amplitude > threshold
        pulse_starts = np.where(np.diff(pulse_mask.astype(int)) > 0)[0]
        
        if len(pulse_starts) < 2:
            return 50e3  # Default value
            
        # Calculate PRF from pulse timing
        pulse_intervals = np.diff(pulse_starts)
        avg_samples_per_pulse = np.mean(pulse_intervals)
        
        return float(fs / avg_samples_per_pulse)
    
    def _class_to_modulation_type(self, waveform_class: WaveformClass) -> ModulationType:
        """Convert waveform class to modulation type."""
        class_to_mod = {
            WaveformClass.AM: ModulationType.AM,
            WaveformClass.FM: ModulationType.FM,
            WaveformClass.PSK: ModulationType.PSK,
            WaveformClass.QAM: ModulationType.QAM,
        }
        return class_to_mod.get(waveform_class, ModulationType.NONE)