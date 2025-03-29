"""Advanced waveform types for the Helios waveform library."""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid

from helios.core.data_structures import Waveform, ModulationType
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class OFDMWaveform(Waveform):
    """Orthogonal Frequency Division Multiplexing waveform."""
    num_subcarriers: int = 64
    cyclic_prefix_length: int = 16
    subcarrier_spacing: float = 15e3  # 15 kHz spacing (typical in LTE)
    pilot_indices: List[int] = field(default_factory=lambda: [0, 16, 32, 48])
    
@dataclass
class SpreadSpectrumWaveform(Waveform):
    """Direct Sequence Spread Spectrum waveform."""
    spreading_code: np.ndarray = field(default_factory=lambda: np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1]))
    chip_rate: float = 1e6  # 1 Mcps
    processing_gain: float = 10.0  # dB
    
@dataclass
class FrequencyHoppingWaveform(Waveform):
    """Frequency hopping waveform."""
    hop_frequencies: List[float] = field(default_factory=list)
    hop_interval: float = 0.001  # seconds
    hop_pattern: str = "random"  # random, sequential, custom
    custom_pattern: Optional[List[int]] = None
    
@dataclass
class ChirpWaveform(Waveform):
    """Linear frequency modulated chirp waveform."""
    start_frequency: float = 0.0  # Hz
    end_frequency: float = 1e6  # Hz
    chirp_rate: Optional[float] = None  # Hz/s, if None calculated from duration
    repeat: bool = False