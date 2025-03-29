"""Tactical waveform definitions for military and defense applications."""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid

from helios.core.data_structures import Waveform, ModulationType
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Link16Waveform(Waveform):
    """Link-16 tactical data link waveform."""
    network_participation_group: int = 1
    time_slots: List[int] = field(default_factory=list)
    frequency_hopping_pattern: str = "standard"  # standard, secure, custom
    transmission_security_level: int = 1
    message_format: str = "JTIDS"  # JTIDS, MIDS, etc.
    pulse_repetition_frequency: float = 50e3  # Hz

@dataclass
class SINCGARSWaveform(Waveform):
    """Single Channel Ground and Airborne Radio System waveform."""
    hop_set: int = 1
    lockout_set: List[float] = field(default_factory=list)  # Frequencies to avoid
    transmission_security_key: str = ""
    hop_rate: float = 100.0  # Hops per second
    mode: str = "frequency_hopping"  # frequency_hopping, single_channel
    
@dataclass
class HaveQuickWaveform(Waveform):
    """HAVE QUICK frequency hopping waveform."""
    word_of_day: str = ""  # Encryption key
    time_of_day: float = 0.0  # Seconds since midnight
    epoch: int = 0
    channel_set: int = 1
    dwell_time: float = 0.01  # seconds per frequency
    
@dataclass
class MILSTDWaveform(Waveform):
    """MIL-STD-188 compliant waveform."""
    standard_type: str = "188-110A"  # 188-110A, 188-110B, etc.
    data_rate: float = 1200.0  # bits per second
    interleaver_type: str = "short"  # short, long, none
    forward_error_correction: bool = True
    robust_mode: bool = False