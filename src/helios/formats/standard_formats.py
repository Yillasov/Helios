"""Standard data exchange formats for Helios interoperability."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

@dataclass
class PositionFormat:
    """Standard position format."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    timestamp: Optional[float] = None

@dataclass
class OrientationFormat:
    """Standard orientation format."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    timestamp: Optional[float] = None

@dataclass
class PlatformStateFormat:
    """Standard format for platform state."""
    platform_id: str
    timestamp: float
    position: PositionFormat
    velocity: Dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vz": 0.0})
    orientation: OrientationFormat = field(default_factory=OrientationFormat)
    custom_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WaveformFormat:
    """Standard format for waveform definition."""
    type: str  # e.g., 'CW', 'Pulse', 'Custom'
    parameters: Dict[str, Any] # e.g., {'pulse_width': 1e-6, 'prf': 1000}
    duration: Optional[float] = None

@dataclass
class SignalFormat:
    """Standard format for signal definition."""
    signal_id: str
    source_platform_id: str
    waveform: WaveformFormat
    frequency: float  # Center frequency in Hz
    bandwidth: float  # Bandwidth in Hz
    power: float      # Power in dBm or Watts (specify units)
    start_time: float
    custom_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HardwareMeasurementFormat:
    """Standard format for hardware measurements."""
    device_id: str
    timestamp: float
    measurement_type: str # e.g., 'IQ', 'Spectrum', 'Power'
    frequency: Optional[float] = None # Center frequency if applicable
    sample_rate: Optional[float] = None # Sample rate if applicable
    data: Any = None  # Could be numpy array path, list of values, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationResultSummaryFormat:
    """Standard format for simulation result summary."""
    run_id: str
    scenario_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    metrics: Dict[str, Union[float, str, list]] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)

# Example Usage (Conceptual)
# platform_state = PlatformStateFormat(...)
# import json
# json_output = json.dumps(dataclasses.asdict(platform_state))