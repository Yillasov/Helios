"""Tactical waveform generator for military communications."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy

from helios.core.data_structures import Waveform, ModulationType
from helios.core.interfaces import IWaveformGenerator
from helios.utils.logger import get_logger
from helios.waveforms.tactical_waveforms import (
    Link16Waveform, SINCGARSWaveform, HaveQuickWaveform, MILSTDWaveform
)
from helios.dsp.modulation import modulate_am, modulate_fm, modulate_psk

logger = get_logger(__name__)

class TacticalWaveformGenerator(IWaveformGenerator):
    """Generator for tactical military waveforms."""
    
    def __init__(self):
        """Initialize the tactical waveform generator."""
        self.time_of_day = 0.0  # Seconds since midnight
        self.crypto_keys = {}  # Store encryption keys
        
    def generate(self, waveform: Waveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples for a tactical waveform."""
        logger.debug(f"Generating tactical waveform: {waveform.id}, duration: {duration}s")
        
        # Dispatch to specialized generators based on waveform type
        if isinstance(waveform, Link16Waveform):
            return self._generate_link16(waveform, sampling_rate, duration)
        elif isinstance(waveform, SINCGARSWaveform):
            return self._generate_sincgars(waveform, sampling_rate, duration)
        elif isinstance(waveform, HaveQuickWaveform):
            return self._generate_have_quick(waveform, sampling_rate, duration)
        elif isinstance(waveform, MILSTDWaveform):
            return self._generate_milstd(waveform, sampling_rate, duration)
        else:
            # Fallback to basic carrier generation
            num_samples = int(sampling_rate * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            carrier = waveform.amplitude * np.exp(2j * np.pi * waveform.center_frequency * t)
            
            # Apply modulation if specified
            if waveform.modulation_type != ModulationType.NONE:
                return self.apply_modulation(carrier, waveform, sampling_rate)
                
            return carrier
    
    def apply_modulation(self, 
                        carrier_samples: np.ndarray, 
                        waveform: Waveform,
                        sampling_rate: float) -> np.ndarray:
        """Apply modulation to a carrier signal."""
        mod_type = waveform.modulation_type
        params = waveform.modulation_params
        
        # Generate message signal if needed
        message = None
        if mod_type in [ModulationType.AM, ModulationType.FM]:
            message = self._generate_message_signal(len(carrier_samples), params, sampling_rate)
        
        if mod_type == ModulationType.AM:
            if message is None:
                raise ValueError("Message signal is required for AM modulation")
            return modulate_am(carrier_samples, message, params.get('modulation_index', 0.5))
        elif mod_type == ModulationType.FM:
            if message is None:
                raise ValueError("Message signal is required for FM modulation")
            return modulate_fm(carrier_samples, message, params.get('modulation_index', 0.5), sampling_rate)
        elif mod_type == ModulationType.PSK:
            bits = np.random.randint(0, 2, params.get('num_bits', 1000))
            symbols = modulate_psk(bits, params.get('bits_per_symbol', 2))
            return self._shape_symbols(symbols, carrier_samples, params)
        else:
            logger.warning(f"Modulation type {mod_type} not implemented, returning carrier")
            return carrier_samples
    
    def _generate_link16(self, waveform: Link16Waveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate Link-16 waveform."""
        logger.debug(f"Generating Link-16 waveform with PRF {waveform.pulse_repetition_frequency} Hz")
        
        # Generate frequency hopping pattern
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Create frequency hopping pattern based on security level
        if waveform.frequency_hopping_pattern == "standard":
            # Simple pseudorandom hopping for demonstration
            hop_interval = 0.0013  # 1.3 ms standard for Link-16
            num_hops = int(duration / hop_interval)
            hop_frequencies = self._generate_link16_hop_pattern(waveform, num_hops)
        else:
            # Default to center frequency
            hop_frequencies = np.ones(num_samples) * waveform.center_frequency
            
        # Generate the signal with frequency hops
        signal = np.zeros(num_samples, dtype=complex)
        hop_samples = int(sampling_rate * 0.0013)  # Use fixed 1.3ms hop interval for Link-16
        
        hop_interval = 0.0013  # Define hop_interval before using it
        num_hops = int(duration / hop_interval)  # Calculate number of hops
        for i in range(min(num_hops, len(hop_frequencies))):
            start_idx = i * hop_samples
            end_idx = min(start_idx + hop_samples, num_samples)
            t_segment = t[start_idx:end_idx]
            freq = hop_frequencies[i]
            signal[start_idx:end_idx] = waveform.amplitude * np.exp(2j * np.pi * freq * t_segment)
        
        return signal
    
    def _generate_sincgars(self, waveform: SINCGARSWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate SINCGARS waveform."""
        logger.debug(f"Generating SINCGARS waveform in {waveform.mode} mode")
        
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        if waveform.mode == "frequency_hopping":
            # Generate frequency hopping pattern
            hop_interval = 1.0 / waveform.hop_rate
            num_hops = int(duration / hop_interval)
            hop_samples = int(sampling_rate * hop_interval)
            
            # Simple pseudorandom hopping based on hop_set
            np.random.seed(waveform.hop_set)  # Use hop_set as seed
            freq_min = waveform.center_frequency - waveform.bandwidth/2
            freq_max = waveform.center_frequency + waveform.bandwidth/2
            hop_frequencies = np.random.uniform(freq_min, freq_max, num_hops)
            
            # Remove lockout frequencies
            for lockout_freq in waveform.lockout_set:
                mask = np.abs(hop_frequencies - lockout_freq) < 25e3  # 25 kHz guard band
                # Fix: Convert np.sum(mask) to int to match the uniform function signature
                hop_frequencies[mask] = np.random.uniform(freq_min, freq_max, int(np.sum(mask)))
            
            # Generate the signal with frequency hops
            signal = np.zeros(num_samples, dtype=complex)
            for i in range(min(num_hops, len(hop_frequencies))):
                start_idx = i * hop_samples
                end_idx = min(start_idx + hop_samples, num_samples)
                t_segment = t[start_idx:end_idx]
                freq = hop_frequencies[i]
                signal[start_idx:end_idx] = waveform.amplitude * np.exp(2j * np.pi * freq * t_segment)
        else:
            # Single channel mode
            signal = waveform.amplitude * np.exp(2j * np.pi * waveform.center_frequency * t)
        
        return signal
    
    def _generate_have_quick(self, waveform: HaveQuickWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate HAVE QUICK waveform."""
        logger.debug(f"Generating HAVE QUICK waveform with dwell time {waveform.dwell_time}s")
        
        # Simple implementation for demonstration
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate frequency hopping pattern based on time of day and word of day
        dwell_samples = int(sampling_rate * waveform.dwell_time)
        num_dwells = int(np.ceil(num_samples / dwell_samples))
        
        # Use hash of word_of_day and time_of_day to seed the random generator
        seed = hash(waveform.word_of_day) + int(waveform.time_of_day)
        np.random.seed(seed)
        
        # Generate frequencies based on channel set
        freq_min = 225e6  # VHF/UHF military band
        freq_max = 400e6
        hop_frequencies = np.random.uniform(freq_min, freq_max, num_dwells)
        
        # Generate the signal with frequency hops
        signal = np.zeros(num_samples, dtype=complex)
        for i in range(num_dwells):
            start_idx = i * dwell_samples
            end_idx = min(start_idx + dwell_samples, num_samples)
            t_segment = t[start_idx:end_idx]
            freq = hop_frequencies[i]
            signal[start_idx:end_idx] = waveform.amplitude * np.exp(2j * np.pi * freq * t_segment)
        
        return signal
    
    def _generate_milstd(self, waveform: MILSTDWaveform, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate MIL-STD-188 compliant waveform."""
        logger.debug(f"Generating MIL-STD-188 {waveform.standard_type} waveform")
        
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate carrier
        carrier = waveform.amplitude * np.exp(2j * np.pi * waveform.center_frequency * t)
        
        # Apply appropriate modulation based on standard type
        if waveform.standard_type == "188-110A":
            # 8-PSK modulation
            bits_per_symbol = 3
            num_bits = int(waveform.data_rate * duration)
            # Generate random data bits
            data_bits = np.random.randint(0, 2, num_bits)
            
            # Apply FEC if enabled
            if waveform.forward_error_correction:
                # Simple repetition code for demonstration
                data_bits = np.repeat(data_bits, 3)
                data_bits = np.reshape(data_bits, (-1, 3))
                data_bits = np.sum(data_bits, axis=1) > 1  # Majority vote
            
            # Apply interleaving if enabled
            if waveform.interleaver_type != "none":
                block_size = 1024 if waveform.interleaver_type == "long" else 256
                # Simple block interleaver
                num_blocks = int(np.ceil(len(data_bits) / block_size))
                for i in range(num_blocks):
                    start_idx = i * block_size
                    end_idx = min(start_idx + block_size, len(data_bits))
                    block = data_bits[start_idx:end_idx]
                    if len(block) == block_size:  # Only interleave complete blocks
                        block = block.reshape(32, -1).T.flatten()
                        data_bits[start_idx:end_idx] = block
            
            # Convert bits to 8-PSK symbols
            symbols = np.zeros(len(data_bits) // bits_per_symbol, dtype=complex)
            for i in range(0, len(data_bits), bits_per_symbol):
                if i + bits_per_symbol <= len(data_bits):
                    symbol_bits = data_bits[i:i+bits_per_symbol]
                    symbol_idx = int(''.join(map(str, symbol_bits)), 2)
                    angle = symbol_idx * 2 * np.pi / 8
                    symbols[i // bits_per_symbol] = np.exp(1j * angle)
            
            # Shape and upsample symbols
            symbol_rate = waveform.data_rate / bits_per_symbol
            samples_per_symbol = int(sampling_rate / symbol_rate)
            modulated = np.repeat(symbols, samples_per_symbol)
            
            # Trim or pad to match carrier length
            if len(modulated) > len(carrier):
                modulated = modulated[:len(carrier)]
            elif len(modulated) < len(carrier):
                modulated = np.pad(modulated, (0, len(carrier) - len(modulated)))
            
            return carrier * modulated
        else:
            # Default to simple PSK for other standards
            return carrier
    
    def _generate_link16_hop_pattern(self, waveform: Link16Waveform, num_hops: int) -> np.ndarray:
        """Generate Link-16 frequency hopping pattern."""
        # Simple pseudorandom pattern for demonstration
        np.random.seed(waveform.network_participation_group + waveform.transmission_security_level)
        
        # Link-16 uses 51 frequencies in the 960-1215 MHz band
        base_freq = 969.0e6  # Base frequency
        step = 3.0e6  # 3 MHz steps
        frequencies = np.array([base_freq + i * step for i in range(51)])
        
        # Select frequencies based on security level
        hop_indices = np.random.choice(len(frequencies), num_hops)
        return frequencies[hop_indices]
    
    def _generate_message_signal(self, length: int, params: Dict[str, Any], sampling_rate: float) -> np.ndarray:
        """Generate message signal for modulation."""
        message_type = params.get('message_type', 'sine')
        message_freq = params.get('message_frequency', 1000.0)
        t = np.linspace(0, length / sampling_rate, length, endpoint=False)
        
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