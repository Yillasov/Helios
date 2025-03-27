"""Utility functions for generating common HPM waveform patterns."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from helios.core.data_structures import HPMWaveform
from helios.waveforms.hpm_generator import HPMWaveformGenerator

def create_high_power_sine_burst(
    center_frequency: float,
    peak_power: float,
    pulse_width: float = 100e-9,  # 100 ns default
    pri: float = 1e-6,  # 1 μs default
    burst_frequency: Optional[float] = None,
    duration: float = 1.0,
    **kwargs
) -> HPMWaveform:
    """
    Create a high-power sine burst waveform.
    
    Args:
        center_frequency: Center frequency in Hz
        peak_power: Peak power in Watts
        pulse_width: Pulse width in seconds
        pri: Pulse repetition interval in seconds
        burst_frequency: Frequency of the sine burst (default: 1/pulse_width)
        duration: Total duration of the waveform in seconds
        **kwargs: Additional parameters for the waveform
        
    Returns:
        Configured HPMWaveform object
    """
    generator = HPMWaveformGenerator()
    
    # Set default burst frequency if not provided
    if burst_frequency is None:
        burst_frequency = 1.0 / pulse_width
    
    # Set pulse shape parameters
    pulse_shape_params = kwargs.pop('pulse_shape_params', {})
    pulse_shape_params['burst_frequency'] = burst_frequency
    
    return generator.generate_hpm_waveform(
        center_frequency=center_frequency,
        bandwidth=2.0 / pulse_width,  # Estimate bandwidth from pulse width
        peak_power=peak_power,
        pulse_width=pulse_width,
        pulse_repetition_interval=pri,
        pulse_shape="sine_burst",
        pulse_shape_params=pulse_shape_params,
        duration=duration,
        **kwargs
    )

def create_square_pulse_hpm(
    center_frequency: float,
    peak_power: float,
    pulse_width: float = 1e-6,  # 1 μs default
    pri: float = 10e-6,  # 10 μs default
    rise_time: float = 10e-9,  # 10 ns rise time
    fall_time: float = 10e-9,  # 10 ns fall time
    duration: float = 1.0,
    **kwargs
) -> HPMWaveform:
    """
    Create a high-power square pulse waveform.
    
    Args:
        center_frequency: Center frequency in Hz
        peak_power: Peak power in Watts
        pulse_width: Pulse width in seconds
        pri: Pulse repetition interval in seconds
        rise_time: Rise time in seconds
        fall_time: Fall time in seconds
        duration: Total duration of the waveform in seconds
        **kwargs: Additional parameters for the waveform
        
    Returns:
        Configured HPMWaveform object
    """
    generator = HPMWaveformGenerator()
    
    return generator.generate_hpm_waveform(
        center_frequency=center_frequency,
        bandwidth=1.0 / rise_time,  # Estimate bandwidth from rise time
        peak_power=peak_power,
        pulse_width=pulse_width,
        pulse_repetition_interval=pri,
        rise_time=rise_time,
        fall_time=fall_time,
        pulse_shape="rectangular",
        duration=duration,
        **kwargs
    )

def create_damped_sine_hpm(
    center_frequency: float,
    peak_power: float,
    pulse_width: float = 200e-9,  # 200 ns default
    pri: float = 2e-6,  # 2 μs default
    damping_factor: Optional[float] = None,  # Will be calculated if None
    oscillation_frequency: Optional[float] = None,  # Will be calculated if None
    duration: float = 1.0,
    **kwargs
) -> HPMWaveform:
    """
    Create a high-power damped sine waveform.
    
    Args:
        center_frequency: Center frequency in Hz
        peak_power: Peak power in Watts
        pulse_width: Pulse width in seconds
        pri: Pulse repetition interval in seconds
        damping_factor: Damping factor (default: 5/pulse_width)
        oscillation_frequency: Oscillation frequency (default: 10/pulse_width)
        duration: Total duration of the waveform in seconds
        **kwargs: Additional parameters for the waveform
        
    Returns:
        Configured HPMWaveform object
    """
    generator = HPMWaveformGenerator()
    
    # Set default parameters if not provided
    if damping_factor is None:
        damping_factor = 5.0 / pulse_width
    
    if oscillation_frequency is None:
        oscillation_frequency = 10.0 / pulse_width
    
    # Set pulse shape parameters
    pulse_shape_params = kwargs.pop('pulse_shape_params', {})
    pulse_shape_params['damping_factor'] = damping_factor
    pulse_shape_params['oscillation_frequency'] = oscillation_frequency
    
    return generator.generate_hpm_waveform(
        center_frequency=center_frequency,
        bandwidth=oscillation_frequency * 2,  # Estimate bandwidth
        peak_power=peak_power,
        pulse_width=pulse_width,
        pulse_repetition_interval=pri,
        pulse_shape="damped_sine",
        pulse_shape_params=pulse_shape_params,
        duration=duration,
        **kwargs
    )

def create_chirp_hpm(
    center_frequency: float,
    peak_power: float,
    pulse_width: float = 500e-9,  # 500 ns default
    pri: float = 5e-6,  # 5 μs default
    start_frequency: Optional[float] = None,  # Will be calculated if None
    end_frequency: Optional[float] = None,  # Will be calculated if None
    duration: float = 1.0,
    **kwargs
) -> HPMWaveform:
    """
    Create a high-power chirp waveform with frequency sweep.
    
    Args:
        center_frequency: Center frequency in Hz
        peak_power: Peak power in Watts
        pulse_width: Pulse width in seconds
        pri: Pulse repetition interval in seconds
        start_frequency: Start frequency of chirp (default: 1/pulse_width)
        end_frequency: End frequency of chirp (default: 10/pulse_width)
        duration: Total duration of the waveform in seconds
        **kwargs: Additional parameters for the waveform
        
    Returns:
        Configured HPMWaveform object
    """
    generator = HPMWaveformGenerator()
    
    # Set default parameters if not provided
    if start_frequency is None:
        start_frequency = 1.0 / pulse_width
    
    if end_frequency is None:
        end_frequency = 10.0 / pulse_width
    
    # Set pulse shape parameters
    pulse_shape_params = kwargs.pop('pulse_shape_params', {})
    pulse_shape_params['start_frequency'] = start_frequency
    pulse_shape_params['end_frequency'] = end_frequency
    
    # Calculate bandwidth from chirp range
    bandwidth = end_frequency - start_frequency
    
    return generator.generate_hpm_waveform(
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        peak_power=peak_power,
        pulse_width=pulse_width,
        pulse_repetition_interval=pri,
        pulse_shape="chirp",
        pulse_shape_params=pulse_shape_params,
        duration=duration,
        **kwargs
    )