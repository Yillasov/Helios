"""Cognitive waveform generation capabilities."""

import numpy as np
from typing import Dict, Any, Optional, List
import copy # Import the copy module

from helios.core.data_structures import Waveform, ModulationType, CognitiveWaveform, AdaptationGoal
from helios.core.interfaces import IWaveformGenerator
from helios.utils.logger import get_logger
from helios.dsp.modulation import modulate_am, modulate_fm, modulate_psk
from helios.dsp.spectral import apply_fft, apply_ifft
from helios.dsp.windows import apply_window
from helios.cognitive.engine import CognitiveEngine
from helios.cognitive.data_structures import EnvironmentState, SpectrumBand, SpectrumOccupancy

logger = get_logger(__name__)

class CognitiveWaveformGenerator(IWaveformGenerator):
    """Advanced waveform generator with cognitive capabilities."""

    def __init__(self):
        """Initialize the cognitive waveform generator."""
        self.sampling_rate = 1e6  # Default sampling rate
        self.current_environment_state: Dict[str, Any] = {} # Placeholder for env feedback
        self.adaptation_memory: Dict[str, Any] = {} # To store history or learned parameters
        
        # Initialize the cognitive engine
        self.cognitive_engine = CognitiveEngine()
        
        # Track the current environment state
        self.env_state = EnvironmentState(timestamp=0.0)

    def update_environment_state(self, state: Dict[str, Any]):
        """Receive updates about the RF environment."""
        self.current_environment_state.update(state)
        logger.debug(f"Cognitive generator received environment update: {state}")
        
        # Convert to structured environment state
        self._update_structured_environment_state(state)
        
        # Update the cognitive engine
        self.cognitive_engine.update_environment_state(self.env_state)

    def _update_structured_environment_state(self, state: Dict[str, Any]):
        """Convert raw environment state to structured EnvironmentState."""
        # Update timestamp if provided
        if 'timestamp' in state:
            self.env_state.timestamp = state['timestamp']
        
        # Update spectrum occupancy if provided
        if 'spectrum_bands' in state:
            for band_info in state['spectrum_bands']:
                band_id = band_info.get('id', f"band_{band_info['start_freq']}_{band_info['end_freq']}")
                
                # Create spectrum band
                band = SpectrumBand(
                    start_freq=band_info['start_freq'],
                    end_freq=band_info['end_freq']
                )
                
                # Create occupancy information
                occupancy = SpectrumOccupancy(
                    band=band,
                    power_level=band_info.get('power_level', -100),
                    noise_floor=band_info.get('noise_floor', -110),
                    occupation_threshold=band_info.get('threshold', 6.0)
                )
                
                # Add to environment state
                self.env_state.spectrum_occupancy[band_id] = occupancy
        
        # Update overall metrics
        if 'overall_noise_level' in state:
            self.env_state.overall_noise_level = state['overall_noise_level']
        
        if 'average_snr' in state:
            self.env_state.average_snr = state['average_snr']
        
        if 'detected_signals' in state:
            self.env_state.detected_signals = state['detected_signals']
        
        # Update interference sources if provided
        if 'interference_sources' in state:
            self.env_state.interference_sources = state['interference_sources']

    def adapt_parameters(self, waveform: CognitiveWaveform) -> Waveform:
        """
        Adapt waveform parameters based on goals, constraints, and environment.
        Now uses the cognitive engine for decision making.
        """
        logger.info(f"Adapting parameters for waveform {waveform.id} based on goals: {waveform.adaptation_goals}")
        
        # Create a copy of the waveform to modify
        adapted_waveform = copy.deepcopy(waveform)
        
        # Get adaptation decisions from cognitive engine
        adaptations = self.cognitive_engine.decide_adaptation(waveform)
        
        # Apply adaptations to the waveform
        if adaptations:
            for param, value in adaptations.items():
                if hasattr(adapted_waveform, param):
                    setattr(adapted_waveform, param, value)
                    logger.info(f"Adapted {param} to {value}")
                else:
                    logger.warning(f"Cannot adapt unknown parameter: {param}")
        
        # Remember the adaptation (optional)
        self.adaptation_memory[waveform.id] = {
            "timestamp": self.env_state.timestamp,
            "adaptations": adaptations,
            "params": {
                "center_frequency": adapted_waveform.center_frequency,
                "bandwidth": adapted_waveform.bandwidth,
                "amplitude": adapted_waveform.amplitude,
                "modulation_type": adapted_waveform.modulation_type
            }
        }
        
        logger.info(f"Adaptation resulted in params: Freq={adapted_waveform.center_frequency/1e6:.2f} MHz, "
                   f"BW={adapted_waveform.bandwidth/1e6:.2f} MHz, Amp={adapted_waveform.amplitude:.2f}")
        
        return adapted_waveform

    def generate(self, waveform: Waveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples for a waveform, adapting if cognitive."""
        logger.debug(f"Generating waveform: {waveform.id}, duration: {duration}s")
        self.sampling_rate = sampling_rate

        # --- Cognitive Adaptation Step ---
        if isinstance(waveform, CognitiveWaveform):
            # Perform adaptation based on current state before generating
            # In a real system, this might happen based on events or timers
            adapted_waveform = self.adapt_parameters(waveform)
        else:
            # Not a cognitive waveform, use as is
            adapted_waveform = waveform
        # --- End Adaptation Step ---


        # Generate base carrier using potentially adapted parameters
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        # Use adapted_waveform parameters here
        carrier = adapted_waveform.amplitude * np.exp(2j * np.pi * adapted_waveform.center_frequency * t)

        # Apply modulation if specified (using adapted parameters)
        if adapted_waveform.modulation_type != ModulationType.NONE:
            return self.apply_modulation(carrier, adapted_waveform)

        return carrier

    # Make sure apply_modulation and helper methods handle Waveform type correctly
    def apply_modulation(self, samples: np.ndarray, waveform: Waveform) -> np.ndarray:
        """Apply modulation to a carrier signal."""
        mod_type = waveform.modulation_type
        params = waveform.modulation_params

        # Generate message signal if needed
        message = None  # Initialize message variable
        if mod_type in [ModulationType.AM, ModulationType.FM, ModulationType.PM]:
            # Pass potentially adapted parameters if needed for message generation
            message = self._generate_message_signal(len(samples), params)

        if mod_type == ModulationType.AM:
            if message is None: message = self._generate_message_signal(len(samples), params)
            return modulate_am(samples, message, params.get('modulation_index', 0.5))
        elif mod_type == ModulationType.FM:
            if message is None: message = self._generate_message_signal(len(samples), params)
            return modulate_fm(samples, message, params.get('modulation_index', 0.5), self.sampling_rate)
        elif mod_type == ModulationType.PSK:
            bits = np.random.randint(0, 2, params.get('num_bits', 1000))
            symbols = modulate_psk(bits, params.get('bits_per_symbol', 2))
            # Pass adapted params if needed by _shape_symbols
            return self._shape_symbols(symbols, samples, params)
        else:
            logger.warning(f"Modulation type {mod_type} not implemented, returning carrier")
            return samples

    def _generate_message_signal(self, length: int, params: Dict[str, Any]) -> np.ndarray:
        """Generate message signal for modulation."""
        message_type = params.get('message_type', 'sine')
        message_freq = params.get('message_frequency', 1000.0)
        t = np.linspace(0, length / self.sampling_rate, length, endpoint=False)
        
        if message_type == 'sine':
            return np.sin(2 * np.pi * message_freq * t)
        elif message_type == 'square':
            return np.sign(np.sin(2 * np.pi * message_freq * t))
        elif message_type == 'triangle':
            return 2 * np.abs(2 * (t * message_freq - np.floor(t * message_freq + 0.5))) - 1
        else:
            return np.random.randn(length)  # Random noise

    def _shape_symbols(self, symbols: np.ndarray, carrier: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Shape symbols with pulse shaping filter."""
        # Simple rectangular pulse shaping
        samples_per_symbol = params.get('samples_per_symbol', 8)
        shaped = np.repeat(symbols, samples_per_symbol)
        
        # Pad or truncate to match carrier length
        if len(shaped) < len(carrier):
            shaped = np.pad(shaped, (0, len(carrier) - len(shaped)))
        else:
            shaped = shaped[:len(carrier)]
        
        return shaped * carrier