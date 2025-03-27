"""High-Power Microwave (HPM) waveform generation capabilities."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import scipy.signal as signal

from helios.core.data_structures import Waveform, PulsedWaveform, HPMWaveform, ModulationType
from helios.core.interfaces import IWaveformGenerator
from helios.utils.logger import get_logger
from helios.waveforms.basic_generator import BasicWaveformGenerator

logger = get_logger(__name__)

class HPMWaveformGenerator(BasicWaveformGenerator):
    """Generator for High-Power Microwave waveforms."""
    
    def __init__(self):
        """Initialize the HPM waveform generator."""
        super().__init__()
        self.pulse_shapes = {
            "rectangular": self._generate_rectangular_pulse,
            "gaussian": self._generate_gaussian_pulse,
            "sine_burst": self._generate_sine_burst,
            "damped_sine": self._generate_damped_sine,
            "chirp": self._generate_chirp_pulse,
            "custom": self._generate_custom_pulse
        }
    
    def generate(self, waveform: Waveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples for a waveform."""
        logger.debug(f"Generating HPM waveform: {waveform.id}, duration: {duration}s")
        
        # Handle pulsed waveforms
        if isinstance(waveform, PulsedWaveform):
            return self._generate_pulsed_waveform(waveform, sampling_rate, duration)
        
        # For non-pulsed waveforms, use the parent class implementation
        return super().generate(waveform, sampling_rate, duration)
    
    def _generate_pulsed_waveform(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate a pulsed waveform."""
        # Calculate number of samples
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate carrier
        carrier = waveform.amplitude * np.exp(2j * np.pi * waveform.center_frequency * t)
        
        # Generate pulse envelope
        pulse_shape = waveform.pulse_shape.lower()
        if pulse_shape in self.pulse_shapes:
            pulse_envelope = self.pulse_shapes[pulse_shape](waveform, sampling_rate, duration)
        else:
            logger.warning(f"Unknown pulse shape: {pulse_shape}, using rectangular")
            pulse_envelope = self.pulse_shapes["rectangular"](waveform, sampling_rate, duration)
        
        # Apply pulse envelope to carrier
        pulsed_signal = carrier * pulse_envelope
        
        # Apply modulation if specified
        if waveform.modulation_type != ModulationType.NONE:
            modulated_signal = self.apply_modulation(pulsed_signal, waveform)
            return modulated_signal
        
        return pulsed_signal
    
    def _generate_rectangular_pulse(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate rectangular pulse envelope."""
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate pulse parameters
        pri = waveform.pulse_repetition_interval
        pw = waveform.pulse_width
        rise_time = waveform.rise_time
        fall_time = waveform.fall_time
        
        # Initialize pulse envelope
        pulse_envelope = np.zeros(num_samples)
        
        # Generate pulses
        for pulse_start in np.arange(0, duration, pri):
            pulse_end = pulse_start + pw
            if pulse_end > duration:
                break
                
            # Convert times to sample indices
            start_idx = int(pulse_start * sampling_rate)
            end_idx = int(pulse_end * sampling_rate)
            rise_samples = max(1, int(rise_time * sampling_rate))
            fall_samples = max(1, int(fall_time * sampling_rate))
            
            # Create pulse with rise and fall times
            if rise_samples > 1:
                rise_indices = np.arange(start_idx, min(start_idx + rise_samples, num_samples))
                if len(rise_indices) > 0:
                    pulse_envelope[rise_indices] = np.linspace(0, 1, len(rise_indices))
            
            # Flat top of pulse
            flat_indices = np.arange(start_idx + rise_samples, min(end_idx - fall_samples, num_samples))
            if len(flat_indices) > 0:
                pulse_envelope[flat_indices] = 1.0
            
            # Fall time
            if fall_samples > 1:
                fall_indices = np.arange(end_idx - fall_samples, min(end_idx, num_samples))
                if len(fall_indices) > 0:
                    pulse_envelope[fall_indices] = np.linspace(1, 0, len(fall_indices))
        
        return pulse_envelope
    
    def _generate_gaussian_pulse(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate Gaussian pulse envelope."""
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate pulse parameters
        pri = waveform.pulse_repetition_interval
        pw = waveform.pulse_width
        
        # Standard deviation of Gaussian (adjust as needed)
        sigma = pw / 6.0  # 6-sigma within the pulse width
        
        # Initialize pulse envelope
        pulse_envelope = np.zeros(num_samples)
        
        # Generate pulses
        for pulse_center in np.arange(pw/2, duration, pri):
            # Gaussian pulse centered at pulse_center
            gaussian = np.exp(-0.5 * ((t - pulse_center) / sigma) ** 2)
            pulse_envelope += gaussian
        
        # Normalize to maximum of 1
        if np.max(pulse_envelope) > 0:
            pulse_envelope = pulse_envelope / np.max(pulse_envelope)
        
        return pulse_envelope
    
    def _generate_sine_burst(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate sine burst pulse envelope."""
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate pulse parameters
        pri = waveform.pulse_repetition_interval
        pw = waveform.pulse_width
        
        # Get burst frequency from parameters or use default
        burst_freq = waveform.pulse_shape_params.get('burst_frequency', 1.0 / pw)
        
        # Initialize pulse envelope
        pulse_envelope = np.zeros(num_samples)
        
        # Generate pulses
        for pulse_start in np.arange(0, duration, pri):
            pulse_end = pulse_start + pw
            if pulse_end > duration:
                break
                
            # Convert times to sample indices
            start_idx = int(pulse_start * sampling_rate)
            end_idx = int(pulse_end * sampling_rate)
            
            # Create sine burst
            if end_idx > start_idx:
                pulse_duration = (end_idx - start_idx) / sampling_rate
                pulse_t = np.linspace(0, pulse_duration, end_idx - start_idx, endpoint=False)
                burst = np.sin(2 * np.pi * burst_freq * pulse_t)
                
                # Apply window to smooth edges
                window = np.hanning(len(burst))
                burst = burst * window
                
                # Add to envelope
                pulse_envelope[start_idx:end_idx] = burst
        
        return pulse_envelope
    
    def _generate_damped_sine(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate damped sine pulse envelope."""
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate pulse parameters
        pri = waveform.pulse_repetition_interval
        pw = waveform.pulse_width
        
        # Get damping factor and frequency from parameters or use defaults
        damping = waveform.pulse_shape_params.get('damping_factor', 5.0 / pw)
        freq = waveform.pulse_shape_params.get('oscillation_frequency', 10.0 / pw)
        
        # Initialize pulse envelope
        pulse_envelope = np.zeros(num_samples)
        
        # Generate pulses
        for pulse_start in np.arange(0, duration, pri):
            if pulse_start + pw > duration:
                break
                
            # Generate damped sine for this pulse
            pulse_t = t - pulse_start
            mask = (pulse_t >= 0) & (pulse_t <= pw)
            damped_sine = np.exp(-damping * pulse_t[mask]) * np.sin(2 * np.pi * freq * pulse_t[mask])
            
            # Normalize this pulse
            if len(damped_sine) > 0 and np.max(np.abs(damped_sine)) > 0:
                damped_sine = damped_sine / np.max(np.abs(damped_sine))
            
            # Add to envelope
            pulse_indices = np.where(mask)[0]
            if len(pulse_indices) > 0:
                pulse_envelope[pulse_indices] = damped_sine
        
        return pulse_envelope
    
    def _generate_chirp_pulse(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate chirp pulse envelope with frequency sweep."""
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Calculate pulse parameters
        pri = waveform.pulse_repetition_interval
        pw = waveform.pulse_width
        
        # Get chirp parameters from pulse_shape_params or use defaults
        f0 = waveform.pulse_shape_params.get('start_frequency', 1.0 / pw)
        f1 = waveform.pulse_shape_params.get('end_frequency', 10.0 / pw)
        
        # Initialize pulse envelope
        pulse_envelope = np.zeros(num_samples)
        
        # Generate pulses
        for pulse_start in np.arange(0, duration, pri):
            pulse_end = pulse_start + pw
            if pulse_end > duration:
                break
                
            # Convert times to sample indices
            start_idx = int(pulse_start * sampling_rate)
            end_idx = int(pulse_end * sampling_rate)
            
            # Create chirp
            if end_idx > start_idx:
                pulse_duration = (end_idx - start_idx) / sampling_rate
                pulse_t = np.linspace(0, pulse_duration, end_idx - start_idx, endpoint=False)
                
                # Linear frequency sweep
                chirp_signal = signal.chirp(pulse_t, f0, pulse_duration, f1)
                
                # Apply window to smooth edges
                window = np.hanning(len(chirp_signal))
                chirp_signal = chirp_signal * window
                
                # Add to envelope
                pulse_envelope[start_idx:end_idx] = chirp_signal
        
        return pulse_envelope
    
    def _generate_custom_pulse(self, waveform: PulsedWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate custom pulse shape from user-provided function."""
        # Check if custom_function is provided in pulse_shape_params
        custom_func = waveform.pulse_shape_params.get('custom_function', None)
        
        if custom_func is None:
            logger.warning("No custom function provided for custom pulse shape, using rectangular")
            return self._generate_rectangular_pulse(waveform, sampling_rate, duration)
        
        try:
            # Call the custom function with the waveform parameters
            return custom_func(waveform, sampling_rate, duration)
        except Exception as e:
            logger.error(f"Error in custom pulse function: {e}")
            return self._generate_rectangular_pulse(waveform, sampling_rate, duration)

    def generate_hpm_waveform(self, 
                             center_frequency: float,
                             bandwidth: float,
                             peak_power: float,
                             pulse_width: float,
                             pulse_repetition_interval: float,
                             pulse_shape: str = "rectangular",
                             duration: float = 1.0,
                             **kwargs) -> HPMWaveform:
        """
        Convenience method to create an HPM waveform with specified parameters.
        
        Args:
            center_frequency: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            peak_power: Peak power in Watts
            pulse_width: Pulse width in seconds
            pulse_repetition_interval: Time between pulses in seconds
            pulse_shape: Shape of the pulse ("rectangular", "gaussian", "sine_burst", etc.)
            duration: Total duration of the waveform in seconds
            **kwargs: Additional parameters for the waveform
            
        Returns:
            Configured HPMWaveform object
        """
        # Convert peak power to amplitude (assuming 50 ohm impedance)
        amplitude = np.sqrt(peak_power * 50)
        
        # Calculate duty cycle
        duty_cycle = pulse_width / pulse_repetition_interval
        
        # Create waveform ID
        waveform_id = kwargs.pop('id', f"hpm_{center_frequency/1e6:.1f}MHz_{peak_power:.1e}W")
        
        # Extract pulse shape parameters
        pulse_shape_params = kwargs.pop('pulse_shape_params', {})
        
        # Create the HPM waveform
        hpm_waveform = HPMWaveform(
            id=waveform_id,
            center_frequency=center_frequency,
            bandwidth=bandwidth,
            amplitude=amplitude,
            pulse_width=pulse_width,
            pulse_repetition_interval=pulse_repetition_interval,
            duty_cycle=duty_cycle,
            pulse_shape=pulse_shape,
            pulse_shape_params=pulse_shape_params,
            peak_power=peak_power,
            duration=duration,
            **kwargs
        )
        
        return hpm_waveform