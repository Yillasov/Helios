"""Agile HPM waveform design tools with combined modulation capabilities."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from helios.core.data_structures import HPMWaveform, ModulationType
from helios.waveforms.hpm_generator import HPMWaveformGenerator
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AgileHPMParameters:
    """Parameters for agile HPM waveform generation."""
    # Frequency agility parameters
    frequency_pattern: str = "fixed"  # fixed, hop, sweep, chirp
    frequency_min: float = 0.0  # Hz
    frequency_max: float = 0.0  # Hz
    frequency_hop_rate: float = 0.0  # Hops per second
    frequency_hop_sequence: List[float] = field(default_factory=list)  # Hz
    
    # Amplitude agility parameters
    amplitude_pattern: str = "fixed"  # fixed, modulated, staggered
    amplitude_min: float = 0.0  # Normalized (0-1)
    amplitude_max: float = 1.0  # Normalized (0-1)
    amplitude_modulation_rate: float = 0.0  # Hz
    
    # Pulse-width agility parameters
    pw_pattern: str = "fixed"  # fixed, staggered, modulated
    pw_min: float = 1e-9  # seconds
    pw_max: float = 1e-6  # seconds
    pw_modulation_rate: float = 0.0  # Hz
    pw_sequence: List[float] = field(default_factory=list)  # seconds
    
    # Combined agility parameters
    agility_sync: bool = False  # Synchronize all agility patterns
    custom_pattern: Optional[Dict[str, Any]] = None  # For custom patterns


class AgileHPMGenerator:
    """Generator for agile HPM waveforms with combined modulation capabilities."""
    
    def __init__(self):
        """Initialize the agile HPM waveform generator."""
        self.base_generator = HPMWaveformGenerator()
    
    def generate_agile_waveform(self, 
                               base_params: Dict[str, Any],
                               agile_params: AgileHPMParameters,
                               sampling_rate: float,
                               duration: float) -> np.ndarray:
        """
        Generate an agile HPM waveform with combined modulation.
        
        Args:
            base_params: Base HPM waveform parameters
            agile_params: Agile modulation parameters
            sampling_rate: Sampling rate in Hz
            duration: Duration in seconds
            
        Returns:
            Complex time-domain samples of the agile waveform
        """
        # Calculate number of samples
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate frequency modulation pattern
        freq_pattern = self._generate_frequency_pattern(agile_params, t)
        
        # Generate amplitude modulation pattern
        amp_pattern = self._generate_amplitude_pattern(agile_params, t)
        
        # Generate pulse-width modulation pattern
        pw_pattern = self._generate_pw_pattern(agile_params, t)
        
        # Generate the agile waveform
        signal = np.zeros(num_samples, dtype=complex)
        
        # Process in segments based on pulse width changes
        current_pos = 0
        while current_pos < num_samples:
            # Get current parameters
            current_time = t[current_pos]
            current_freq = freq_pattern[current_pos]
            current_amp = amp_pattern[current_pos]
            current_pw = pw_pattern[current_pos]
            
            # Calculate segment length based on pulse width
            segment_samples = min(int(current_pw * sampling_rate), num_samples - current_pos)
            
            # Create temporary HPM waveform with current parameters
            temp_waveform = HPMWaveform(
                center_frequency=current_freq,
                amplitude=current_amp,
                pulse_width=current_pw,
                pulse_repetition_interval=base_params.get("pulse_repetition_interval", current_pw * 10),
                peak_power=base_params.get("peak_power", 1e6),
                pulse_shape=base_params.get("pulse_shape", "rectangular")
            )
            
            # Generate segment
            segment = self.base_generator.generate(temp_waveform, sampling_rate, current_pw)
            
            # Ensure segment fits in the remaining signal
            segment_len = min(len(segment), num_samples - current_pos)
            signal[current_pos:current_pos + segment_len] = segment[:segment_len]
            
            # Move to next position
            current_pos += segment_len
        
        return signal
    
    def _generate_frequency_pattern(self, params: AgileHPMParameters, t: np.ndarray) -> np.ndarray:
        """Generate frequency pattern based on agility parameters."""
        if params.frequency_pattern == "fixed":
            return np.ones_like(t) * params.frequency_min
    
        elif params.frequency_pattern == "hop":
            # Handle zero hop rate - return fixed frequency
            if params.frequency_hop_rate <= 0:
                logger.warning("Frequency hop rate is zero or negative for 'hop' pattern. Using fixed frequency_min.")
                return np.ones_like(t) * params.frequency_min
    
            if not params.frequency_hop_sequence:
                # Generate random hop sequence if not provided
                hop_count = max(1, int(params.frequency_hop_rate * t[-1]))
                hop_freqs_np = np.random.uniform(
                    params.frequency_min,
                    params.frequency_max,
                    hop_count
                )
            else:
                # Ensure hop_freqs is a NumPy array
                hop_freqs_np = np.array(params.frequency_hop_sequence, dtype=float)
                if hop_freqs_np.size == 0:
                    logger.warning("Frequency hop sequence is empty. Using fixed frequency_min.")
                    return np.ones_like(t) * params.frequency_min
            
            # Calculate hop duration and indices
            hop_duration = 1.0 / params.frequency_hop_rate
            hop_indices = np.floor(t / hop_duration).astype(int)
            # Clip indices to be within the bounds of the hop_freqs array
            hop_indices = np.clip(hop_indices, 0, len(hop_freqs_np) - 1)
            
            # Index the NumPy array
            return hop_freqs_np[hop_indices]
    
        elif params.frequency_pattern == "sweep":
            # Handle zero or negative rate for sweep (treat as fixed)
            sweep_rate = params.frequency_hop_rate # Re-using hop_rate for sweep speed
            if sweep_rate <= 0:
                 logger.warning("Frequency sweep rate (using frequency_hop_rate) is zero or negative. Using fixed frequency_min.")
                 return np.ones_like(t) * params.frequency_min
            # Linear frequency sweep
            sweep_period = 1.0 / sweep_rate
            phase = (t % sweep_period) / sweep_period
            return params.frequency_min + phase * (params.frequency_max - params.frequency_min)
        
        elif params.frequency_pattern == "chirp":
            # Exponential chirp
            return params.frequency_min * np.exp(
                t * np.log(params.frequency_max / params.frequency_min) / t[-1]
            )
        
        # Default case if pattern is unknown
        logger.warning(f"Unknown frequency pattern: {params.frequency_pattern}. Using fixed frequency_min.")
        return np.ones_like(t) * params.frequency_min
    
        return np.ones_like(t) * params.frequency_min
    
    def _generate_amplitude_pattern(self, params: AgileHPMParameters, t: np.ndarray) -> np.ndarray:
        """Generate amplitude pattern based on agility parameters."""
        if params.amplitude_pattern == "fixed":
            return np.ones_like(t) * params.amplitude_max
        
        elif params.amplitude_pattern == "modulated":
            # Sinusoidal amplitude modulation
            mod_freq = params.amplitude_modulation_rate
            mod = 0.5 * (1 + np.cos(2 * np.pi * mod_freq * t))
            return params.amplitude_min + mod * (params.amplitude_max - params.amplitude_min)
        
        elif params.amplitude_pattern == "staggered":
            # Staggered amplitude pattern
            stagger_period = 1.0 / params.amplitude_modulation_rate
            stagger_phase = (t % stagger_period) / stagger_period
            stagger_levels = np.array([0.25, 0.5, 0.75, 1.0])
            stagger_indices = np.floor(stagger_phase * len(stagger_levels)).astype(int)
            
            return params.amplitude_min + stagger_levels[stagger_indices] * (
                params.amplitude_max - params.amplitude_min
            )
        
        return np.ones_like(t) * params.amplitude_max
    
    def _generate_pw_pattern(self, params: AgileHPMParameters, t: np.ndarray) -> np.ndarray:
        """Generate pulse width pattern based on agility parameters."""
        if params.pw_pattern == "fixed":
            return np.ones_like(t) * params.pw_max
        
        elif params.pw_pattern == "staggered":
            if not params.pw_sequence:
                # Default staggered sequence
                pw_sequence = np.array([0.25, 0.5, 0.75, 1.0]) * (params.pw_max - params.pw_min) + params.pw_min
            else:
                pw_sequence = np.array(params.pw_sequence)
            
            # Generate staggered pattern
            stagger_period = 1.0 / params.pw_modulation_rate if params.pw_modulation_rate > 0 else t[-1]
            stagger_phase = (t % stagger_period) / stagger_period
            stagger_indices = np.floor(stagger_phase * len(pw_sequence)).astype(int)
            
            return pw_sequence[stagger_indices]
        
        elif params.pw_pattern == "modulated":
            # Sinusoidal pulse width modulation
            mod_freq = params.pw_modulation_rate
            mod = 0.5 * (1 + np.cos(2 * np.pi * mod_freq * t))
            return params.pw_min + mod * (params.pw_max - params.pw_min)
        
        return np.ones_like(t) * params.pw_max


def create_agile_hpm_waveform(
    center_frequency: float,
    peak_power: float,
    agile_params: Optional[AgileHPMParameters] = None,
    **kwargs
) -> HPMWaveform:
    """
    Create an agile HPM waveform with combined modulation capabilities.
    
    Args:
        center_frequency: Base center frequency in Hz
        peak_power: Peak power in Watts
        agile_params: Agile modulation parameters
        **kwargs: Additional parameters for the waveform
        
    Returns:
        Configured HPMWaveform object with agile capabilities
    """
    # Create default agile parameters if not provided
    if agile_params is None:
        agile_params = AgileHPMParameters(
            frequency_min=center_frequency,
            frequency_max=center_frequency * 1.1
        )
    
    # Create base waveform
    waveform = HPMWaveform(
        center_frequency=center_frequency,
        peak_power=peak_power,
        pulse_width=kwargs.get('pulse_width', 100e-9),
        pulse_repetition_interval=kwargs.get('pri', 1e-6),
        pulse_shape=kwargs.get('pulse_shape', 'rectangular'),
        **{k: v for k, v in kwargs.items() if k not in ['pulse_width', 'pri', 'pulse_shape']}
    )
    
    # Store agile parameters in pulse_shape_params
    waveform.pulse_shape_params['agile_params'] = {
        'frequency_pattern': agile_params.frequency_pattern,
        'frequency_min': agile_params.frequency_min,
        'frequency_max': agile_params.frequency_max,
        'frequency_hop_rate': agile_params.frequency_hop_rate,
        'amplitude_pattern': agile_params.amplitude_pattern,
        'amplitude_min': agile_params.amplitude_min,
        'amplitude_max': agile_params.amplitude_max,
        'amplitude_modulation_rate': agile_params.amplitude_modulation_rate,
        'pw_pattern': agile_params.pw_pattern,
        'pw_min': agile_params.pw_min,
        'pw_max': agile_params.pw_max,
        'pw_modulation_rate': agile_params.pw_modulation_rate,
        'agility_sync': agile_params.agility_sync
    }
    
    return waveform