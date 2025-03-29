"""Cognitive waveform generation capabilities."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import copy # Import the copy module

from helios.core.data_structures import Waveform, ModulationType, CognitiveWaveform, AdaptationGoal
from helios.core.interfaces import IWaveformGenerator
from helios.utils.logger import get_logger
from helios.dsp.modulation import modulate_am, modulate_fm, modulate_psk
from helios.dsp.spectral import apply_fft, apply_ifft
from helios.dsp.windows import apply_window
from helios.cognitive.engine import CognitiveEngine
from helios.cognitive.data_structures import EnvironmentState, SpectrumBand, SpectrumOccupancy

# Add these imports for tactical waveforms
from helios.waveforms.tactical_waveforms import (
    Link16Waveform, SINCGARSWaveform, HaveQuickWaveform, MILSTDWaveform
)

logger = get_logger(__name__)

# Add these imports at the top of the file
from helios.waveforms.advanced_waveforms import (
    OFDMWaveform, SpreadSpectrumWaveform, FrequencyHoppingWaveform, ChirpWaveform
)

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
        
        # Register specialized waveform generators
        self.specialized_generators = {
            OFDMWaveform: self._generate_ofdm,
            SpreadSpectrumWaveform: self._generate_spread_spectrum,
            FrequencyHoppingWaveform: self._generate_frequency_hopping,
            ChirpWaveform: self._generate_chirp
        }

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

    def generate(self, waveform: Waveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples for a waveform, adapting if cognitive."""
        logger.debug(f"Generating waveform: {waveform.id}, duration: {duration}s")
        self.sampling_rate = sampling_rate

        # --- Cognitive Adaptation Step ---
        if isinstance(waveform, CognitiveWaveform):
            # Perform adaptation based on current state before generating
            adapted_waveform = self.adapt_parameters(waveform)
        else:
            # Not a cognitive waveform, use as is
            adapted_waveform = waveform
        # --- End Adaptation Step ---

        # Check if we have a specialized generator for this waveform type
        for waveform_type, generator in self.specialized_generators.items():
            if isinstance(adapted_waveform, waveform_type):
                return generator(adapted_waveform, sampling_rate, duration)

        # Generate base carrier using potentially adapted parameters
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        carrier = adapted_waveform.amplitude * np.exp(2j * np.pi * adapted_waveform.center_frequency * t)

        # Apply modulation if specified (using adapted parameters)
        if adapted_waveform.modulation_type != ModulationType.NONE:
            return self.apply_modulation(carrier, adapted_waveform, sampling_rate)  # Pass sampling_rate here

        return carrier
    
    # New specialized waveform generators
    def _generate_ofdm(self, waveform: OFDMWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate OFDM waveform samples."""
        logger.debug(f"Generating OFDM waveform with {waveform.num_subcarriers} subcarriers")
        
        # Calculate number of OFDM symbols that fit in the duration
        symbol_duration = waveform.num_subcarriers / sampling_rate
        cp_duration = waveform.cyclic_prefix_length / sampling_rate
        total_symbol_duration = symbol_duration + cp_duration
        num_symbols = max(1, int(duration / total_symbol_duration))
        
        # Generate random data for each subcarrier (QPSK modulation)
        output_signal = np.array([], dtype=complex)
        
        for _ in range(num_symbols):
            # Generate random QPSK symbols for each subcarrier
            data = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j], waveform.num_subcarriers)
            
            # Set pilot subcarriers
            for idx in waveform.pilot_indices:
                if idx < len(data):
                    data[idx] = 1+0j  # Pilot tone
            
            # Perform IFFT to get time domain signal
            time_signal = np.fft.ifft(data) * np.sqrt(waveform.num_subcarriers)
            
            # Add cyclic prefix
            cp = time_signal[-waveform.cyclic_prefix_length:]
            symbol_with_cp = np.concatenate([cp, time_signal])
            
            # Append to output signal
            output_signal = np.concatenate([output_signal, symbol_with_cp])
        
        # Trim or pad to match requested duration
        requested_samples = int(sampling_rate * duration)
        if len(output_signal) > requested_samples:
            output_signal = output_signal[:requested_samples]
        elif len(output_signal) < requested_samples:
            padding = np.zeros(requested_samples - len(output_signal), dtype=complex)
            output_signal = np.concatenate([output_signal, padding])
        
        # Apply amplitude scaling
        output_signal *= waveform.amplitude
        
        # Frequency shift to center frequency
        t = np.arange(len(output_signal)) / sampling_rate
        carrier = np.exp(2j * np.pi * waveform.center_frequency * t)
        return output_signal * carrier
    
    def _generate_spread_spectrum(self, waveform: SpreadSpectrumWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate Direct Sequence Spread Spectrum waveform."""
        logger.debug(f"Generating DSSS waveform with processing gain {waveform.processing_gain} dB")
        
        # Calculate number of data bits that fit in the duration
        chips_per_bit = len(waveform.spreading_code)
        chip_duration = 1.0 / waveform.chip_rate
        bit_duration = chip_duration * chips_per_bit
        num_bits = max(1, int(duration / bit_duration))
        
        # Generate random data bits
        data_bits = np.random.choice([0, 1], num_bits)
        
        # Spread the data bits with the spreading code
        spread_signal = np.array([], dtype=float)
        for bit in data_bits:
            # BPSK modulation: 0 -> -1, 1 -> 1
            bpsk_symbol = 2 * bit - 1
            # Multiply by spreading code
            chip_sequence = bpsk_symbol * waveform.spreading_code
            spread_signal = np.concatenate([spread_signal, chip_sequence])
        
        # Interpolate to match sampling rate
        samples_per_chip = int(sampling_rate / waveform.chip_rate)
        interpolated_signal = np.repeat(spread_signal, samples_per_chip)
        
        # Trim or pad to match requested duration
        requested_samples = int(sampling_rate * duration)
        if len(interpolated_signal) > requested_samples:
            interpolated_signal = interpolated_signal[:requested_samples]
        elif len(interpolated_signal) < requested_samples:
            padding = np.zeros(requested_samples - len(interpolated_signal))
            interpolated_signal = np.concatenate([interpolated_signal, padding])
        
        # Convert to complex baseband and apply amplitude
        baseband_signal = interpolated_signal * waveform.amplitude
        
        # Frequency shift to center frequency
        t = np.arange(len(baseband_signal)) / sampling_rate
        carrier = np.exp(2j * np.pi * waveform.center_frequency * t)
        return baseband_signal * carrier
    
    def _generate_frequency_hopping(self, waveform: FrequencyHoppingWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate Frequency Hopping Spread Spectrum waveform."""
        logger.debug(f"Generating FHSS waveform with hop interval {waveform.hop_interval}s")
        
        # Ensure we have hop frequencies
        if not waveform.hop_frequencies:
            # Default: 10 frequencies around center frequency
            bandwidth = waveform.bandwidth if waveform.bandwidth > 0 else 1e6
            num_hops = 10
            waveform.hop_frequencies = [
                waveform.center_frequency + (i - num_hops/2) * bandwidth/num_hops
                for i in range(num_hops)
            ]
        
        # Calculate number of hops
        num_hops = int(duration / waveform.hop_interval) + 1
        samples_per_hop = int(sampling_rate * waveform.hop_interval)
        
        # Determine hop sequence
        if waveform.hop_pattern == "random":
            hop_sequence = np.random.choice(len(waveform.hop_frequencies), num_hops)
        elif waveform.hop_pattern == "sequential":
            hop_sequence = np.arange(num_hops) % len(waveform.hop_frequencies)
        elif waveform.hop_pattern == "custom" and waveform.custom_pattern is not None:
            # Repeat custom pattern as needed
            pattern = waveform.custom_pattern
            hop_sequence = np.array([pattern[i % len(pattern)] for i in range(num_hops)])
        else:
            # Default to random
            hop_sequence = np.random.choice(len(waveform.hop_frequencies), num_hops)
        
        # Generate signal
        output_signal = np.array([], dtype=complex)
        
        for hop_idx in hop_sequence:
            # Get frequency for this hop
            freq = waveform.hop_frequencies[hop_idx]
            
            # Generate simple tone for this hop
            t = np.arange(samples_per_hop) / sampling_rate
            hop_signal = waveform.amplitude * np.exp(2j * np.pi * (freq - waveform.center_frequency) * t)
            
            # Add to output
            output_signal = np.concatenate([output_signal, hop_signal])
        
        # Trim to requested duration
        requested_samples = int(sampling_rate * duration)
        if len(output_signal) > requested_samples:
            output_signal = output_signal[:requested_samples]
        elif len(output_signal) < requested_samples:
            padding = np.zeros(requested_samples - len(output_signal), dtype=complex)
            output_signal = np.concatenate([output_signal, padding])
        
        # Frequency shift to center frequency
        t = np.arange(len(output_signal)) / sampling_rate
        carrier = np.exp(2j * np.pi * waveform.center_frequency * t)
        return output_signal * carrier
    
    def _generate_chirp(self, waveform: ChirpWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate chirp waveform."""
        logger.debug(f"Generating chirp from {waveform.start_frequency/1e6:.2f} MHz to {waveform.end_frequency/1e6:.2f} MHz")
        
        # Calculate chirp rate if not specified
        if waveform.chirp_rate is None:
            chirp_rate = (waveform.end_frequency - waveform.start_frequency) / duration
        else:
            chirp_rate = waveform.chirp_rate
        
        # Generate time vector
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate chirp: phase is integral of frequency
        # For linear chirp: f(t) = f0 + kt where k is chirp rate
        # Phase: phi(t) = 2π∫f(t)dt = 2π(f0*t + k*t²/2)
        phase = 2 * np.pi * (waveform.start_frequency * t + 0.5 * chirp_rate * t**2)
        
        # Generate complex baseband signal
        chirp_signal = waveform.amplitude * np.exp(1j * phase)
        
        # If repeating, reshape and repeat
        if waveform.repeat and waveform.duration is not None and waveform.duration > 0:
            # Calculate samples for one chirp
            chirp_samples = int(sampling_rate * waveform.duration)
            if chirp_samples < num_samples:
                # Generate one chirp
                one_chirp = chirp_signal[:chirp_samples]
                # Repeat to fill duration
                num_repeats = int(np.ceil(num_samples / chirp_samples))
                repeated = np.tile(one_chirp, num_repeats)
                chirp_signal = repeated[:num_samples]
        
        return chirp_signal

    # Enhanced adaptation strategies
    def adapt_parameters(self, waveform: CognitiveWaveform) -> Waveform:
        """
        Adapt waveform parameters based on goals, constraints, and environment.
        Now with enhanced adaptation strategies.
        """
        logger.info(f"Adapting parameters for waveform {waveform.id} based on goals: {waveform.adaptation_goals}")
        
        # Create a copy of the waveform to modify
        adapted_waveform = copy.deepcopy(waveform)
        
        # Get adaptation decisions from cognitive engine
        adaptations = self.cognitive_engine.decide_adaptation(waveform)
        
        # Enhanced adaptation strategies based on environment
        if not adaptations and self.env_state.detected_signals:
            # Implement interference avoidance if no specific adaptations
            adaptations = self._avoid_interference(waveform)
        
        # Apply adaptations to the waveform
        if adaptations:
            for param, value in adaptations.items():
                if hasattr(adapted_waveform, param):
                    setattr(adapted_waveform, param, value)
                    logger.info(f"Adapted {param} to {value}")
                else:
                    logger.warning(f"Cannot adapt unknown parameter: {param}")
        
        # Remember the adaptation
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
    
    def _avoid_interference(self, waveform: CognitiveWaveform) -> Dict[str, Any]:
        """Implement interference avoidance strategy."""
        # Find occupied bands
        occupied_bands = []
        for band_id, occupancy in self.env_state.spectrum_occupancy.items():
            # Fix: is_occupied is a property, not a method - don't call it with ()
            if occupancy.is_occupied:  # Removed the parentheses
                occupied_bands.append((occupancy.band.start_freq, occupancy.band.end_freq))
        
        # Find a clear frequency if current one is occupied
        current_band = (waveform.center_frequency - waveform.bandwidth/2, 
                        waveform.center_frequency + waveform.bandwidth/2)
        
        for start, end in occupied_bands:
            # Check if current band overlaps with occupied band
            if (current_band[0] <= end and current_band[1] >= start):
                # Find a clear band
                clear_freq = self._find_clear_frequency(occupied_bands, waveform.bandwidth)
                if clear_freq:
                    return {"center_frequency": clear_freq}
        
        # No interference or couldn't find clear frequency
        return {}
    
    # Fix the _find_clear_frequency method to ensure it's using the correct types
    def _find_clear_frequency(self, occupied_bands: List[Tuple[float, float]], 
                             bandwidth: float) -> Optional[float]:
        """Find a clear frequency band."""
        # Sort occupied bands by start frequency
        occupied_bands.sort()
        
        # Check for space before first band
        if occupied_bands and occupied_bands[0][0] > bandwidth:
            return bandwidth/2
        
        # Check for spaces between bands
        for i in range(len(occupied_bands) - 1):
            gap_start = occupied_bands[i][1]
            gap_end = occupied_bands[i+1][0]
            if gap_end - gap_start > bandwidth:
                return (gap_start + gap_end) / 2
        
        # Check for space after last band
        if occupied_bands:
            last_end = occupied_bands[-1][1]
            # Arbitrary upper limit (e.g., 6 GHz)
            if 6e9 - last_end > bandwidth:
                return last_end + bandwidth/2
        
        # Couldn't find clear space
        return None

    # Make sure apply_modulation and helper methods handle Waveform type correctly
    def apply_modulation(self, 
                        carrier_samples: np.ndarray,  # Renamed from samples to carrier_samples
                        waveform: Waveform,
                        sampling_rate: Optional[float] = None) -> np.ndarray:  # Added sampling_rate parameter
        """Apply modulation to a carrier signal."""
        mod_type = waveform.modulation_type
        params = waveform.modulation_params
        
        # Use the sampling_rate parameter or fall back to the instance variable
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
    
        # Generate message signal if needed
        message = None  # Initialize message variable
        if mod_type in [ModulationType.AM, ModulationType.FM, ModulationType.PM]:
            # Pass potentially adapted parameters if needed for message generation
            message = self._generate_message_signal(len(carrier_samples), params)
    
        if mod_type == ModulationType.AM:
            if message is None: message = self._generate_message_signal(len(carrier_samples), params)
            return modulate_am(carrier_samples, message, params.get('modulation_index', 0.5))
        elif mod_type == ModulationType.FM:
            if message is None: message = self._generate_message_signal(len(carrier_samples), params)
            return modulate_fm(carrier_samples, message, params.get('modulation_index', 0.5), sampling_rate)
        elif mod_type == ModulationType.PSK:
            bits = np.random.randint(0, 2, params.get('num_bits', 1000))
            symbols = modulate_psk(bits, params.get('bits_per_symbol', 2))
            # Pass adapted params if needed by _shape_symbols
            return self._shape_symbols(symbols, carrier_samples, params)
        else:
            logger.warning(f"Modulation type {mod_type} not implemented, returning carrier")
            return carrier_samples

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