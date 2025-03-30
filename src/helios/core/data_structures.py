import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any, Callable # Removed duplicate Optional, List, Tuple, Union
import numpy as np

from helios.ecm.techniques import ECMParameters # <<< Add this import

# --- Enumerations ---

class ModulationType(Enum):
    NONE = auto()
    AM = auto()
    FM = auto()
    PM = auto() # Phase Modulation
    PSK = auto() # Phase Shift Keying
    FSK = auto() # Frequency Shift Keying
    QAM = auto() # Quadrature Amplitude Modulation
    # Add more cognitive-specific or complex types later
    OFDM = auto() # Orthogonal Frequency Division Multiplexing

class AdaptationGoal(Enum):
    """Defines the objective for waveform adaptation."""
    MINIMIZE_INTERFERENCE = auto()
    MAXIMIZE_SNR = auto()
    MAXIMIZE_DATA_RATE = auto()
    MINIMIZE_PROBABILITY_OF_INTERCEPT = auto()
    # Add more goals as needed

# --- Basic Geometric & Physical Types ---

@dataclass
class Position:
    """3D position in space (meters)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return np.sqrt((self.x - other.x)**2 +
                       (self.y - other.y)**2 +
                       (self.z - other.z)**2)

@dataclass
class Velocity:
    """3D velocity vector (meters/second)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def magnitude(self) -> float:
        """Calculate the magnitude of the velocity vector."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

@dataclass
class Orientation:
    """3D orientation in Euler angles (radians)."""
    roll: float = 0.0    # Rotation around x-axis
    pitch: float = 0.0   # Rotation around y-axis
    yaw: float = 0.0     # Rotation around z-axis

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        # Implementation of rotation matrix conversion
        # This is a simplified placeholder
        return np.eye(3) # Placeholder: returns identity matrix

# --- Core RF & Simulation Structures ---

@dataclass
class Waveform:
    """Base class for RF waveforms."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    center_frequency: float = 0.0  # Hz
    bandwidth: float = 0.0  # Hz
    amplitude: float = 1.0  # V or relevant unit
    duration: Optional[float] = None # Duration in seconds
    modulation_type: str = "CW" # e.g., CW, AM, FM, PSK, QAM
    modulation_params: Dict[str, Any] = field(default_factory=dict)
    ecm_params: Optional[ECMParameters] = None # This line now has a valid type hint

    @property
    def end_time(self) -> Optional[float]:
        # Placeholder for waveform end time calculation if needed
        return None

    def sample(self, t: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """Sample the waveform at time t. Placeholder."""
        # Basic CW signal
        phase = 2 * np.pi * self.center_frequency * t
        return self.amplitude * np.exp(1j * phase)

    def get_power_spectral_density(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density. Placeholder."""
        # Simplified: box shape centered at center_frequency
        if self.bandwidth > 0:
            freqs = np.linspace(self.center_frequency - self.bandwidth / 2,
                                self.center_frequency + self.bandwidth / 2, 100)
            psd = (self.amplitude ** 2 / self.bandwidth) * np.ones_like(freqs) # Simplified power density
        else:
            freqs = np.array([self.center_frequency])
            psd = np.array([self.amplitude ** 2]) # Simplified total power for CW
        return freqs, psd

    def generate_samples(self, sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples of the waveform.

        Args:
            sampling_rate: Samples per second (Hz)
            duration: Duration to generate (seconds)

        Returns:
            Complex numpy array of waveform samples
        """
        # Base implementation returns a simple carrier wave
        # Note: This might be overridden or unused if a dedicated generator is used
        num_samples = int(sampling_rate * duration)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        return self.amplitude * np.exp(2j * np.pi * self.center_frequency * t)

@dataclass
class CognitiveWaveform(Waveform):
    """A waveform capable of adapting its parameters based on goals and environment."""
    adaptation_goals: List[AdaptationGoal] = field(default_factory=list)
    adaptation_constraints: Dict[str, Any] = field(default_factory=dict) # e.g., {'max_bandwidth': 10e6, 'allowed_frequencies': [(1e9, 1.1e9)]}
    feedback_metrics: List[str] = field(default_factory=lambda: ['snr', 'interference_level']) # Metrics needed for adaptation
    adaptation_strategy: str = "default_reactive" # Identifier for the adaptation logic

    # Add specific cognitive parameters if needed, e.g., learning rate
    learning_rate: Optional[float] = None

from .data_structures import Position, Velocity # Forward reference if needed or adjust imports

@dataclass
class Signal:
    """Represents an RF signal instance in the environment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # ID of the platform or entity that generated this signal
    waveform: Waveform = field(default_factory=lambda: Waveform())
    origin: Position = field(default_factory=Position)  # Where the signal originated
    source_velocity: Velocity = field(default_factory=Velocity) # Velocity of the source at emission time (added)
    emission_time: float = 0.0  # Simulation time when signal starts emission
    direction: Optional[Tuple[float, float]] = None  # Azimuth and elevation if directional
    power: float = 0.0  # Transmit power in dBm
    polarization: str = "vertical"  # vertical, horizontal, circular_lhcp, circular_rhcp, etc.
    propagation_delay: float = 0.0 # Propagation delay in seconds
    doppler_shift: float = 0.0 # Doppler shift in Hz (added)

    def power_at_distance(self, distance: float, frequency: float) -> float:
        """Calculate signal power at a given distance using free space path loss.

        Args:
            distance: Distance in meters
            frequency: Frequency in Hz (used for FSPL calculation)

        Returns:
            Received power in dBm
        """
        if distance <= 1e-6: # Avoid log(0) or division by zero; return original power if very close
            return self.power

        # Ensure frequency is positive to avoid issues with wavelength calculation
        if frequency <= 0:
            # Handle error or return a default value, e.g., negative infinity dBm
            # Or use the waveform's center frequency as a fallback
            frequency = self.waveform.center_frequency
            if frequency <= 0:
                return -np.inf # Indicate an error condition

        # Free space path loss (FSPL) in dB
        # FSPL = 20 * log10(d) + 20 * log10(f) + 20 * log10(4*pi/c)
        # 20 * log10(4*pi/c) approximately equals -27.55 for f in MHz and d in km
        # Using f in Hz and d in meters:
        # FSPL = 20 * log10(distance) + 20 * log10(frequency) - 147.55
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55

        # Received Power = Transmitted Power - Path Loss
        received_power = self.power - fspl_db
        return received_power

@dataclass
class System:
    """Base class for systems that can be equipped on platforms (e.g., radar, comms)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    system_type: str = ""  # E.g., 'radar', 'communication', 'jammer', 'sensor'
    parameters: Dict = field(default_factory=dict) # System-specific parameters

@dataclass
class Platform:
    """Represents entities in the simulation (e.g., aircraft, ground vehicle, satellite)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: Position = field(default_factory=Position)
    velocity: Velocity = field(default_factory=Velocity)
    orientation: Orientation = field(default_factory=Orientation)
    equipped_systems: Dict[str, System] = field(default_factory=dict)
    rcs: Optional[Union[float, Dict[str, float]]] = None  # Radar Cross Section (m^2). Can be simple float or dict for aspect/frequency dependence.
    # Add these new fields for signal processing
    received_signals: Dict[str, Signal] = field(default_factory=dict)  # Signals received by this platform
    combined_signal_power: float = float('-inf')  # Combined power of all received signals in dBm

    def update_position(self, time_delta: float) -> None:
        """Update position based on current velocity and time delta.

        Args:
            time_delta: Time step in seconds
        """
        self.position.x += self.velocity.x * time_delta
        self.position.y += self.velocity.y * time_delta
        self.position.z += self.velocity.z * time_delta

    def add_system(self, system: System) -> None:
        """Add a system to the platform."""
        self.equipped_systems[system.id] = system

    def remove_system(self, system_id: str) -> None:
        """Remove a system from the platform."""
        if system_id in self.equipped_systems:
            del self.equipped_systems[system_id]

@dataclass
class EnvironmentParameters:
    """Defines global environmental characteristics for the simulation."""
    temperature: float = 290.0  # Ambient temperature in Kelvin (default ~17°C)
    noise_floor_density: float = -174.0  # Noise power spectral density in dBm/Hz (kTB at T=290K)
    # Add other relevant parameters as needed, e.g., atmospheric pressure, humidity, terrain model path
    # terrain_model: Optional[str] = None
    # atmospheric_model: Optional[str] = None

    def calculate_noise_power(self, bandwidth: float) -> float:
        """Calculate thermal noise power for a given bandwidth.

        Args:
            bandwidth: Bandwidth in Hz

        Returns:
            Noise power in dBm
        """
        if bandwidth <= 0:
            return -np.inf # Noise power is undefined for non-positive bandwidth

        # Noise Power (dBm) = Noise Floor Density (dBm/Hz) + 10 * log10(Bandwidth (Hz))
        return self.noise_floor_density + 10 * np.log10(bandwidth)

@dataclass
class Scenario:
    """Container for all elements defining a simulation scenario."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    platforms: Dict[str, Platform] = field(default_factory=dict)
    initial_signals: List[Signal] = field(default_factory=list) # Signals present at t=0 or scheduled
    environment: EnvironmentParameters = field(default_factory=EnvironmentParameters)
    start_time: float = 0.0      # Simulation start time (seconds)
    duration: float = 3600.0   # Total simulation duration (seconds)
    # Note: time_step might be better handled by the simulation engine itself

    def add_platform(self, platform: Platform) -> None:
        """Add a platform to the scenario."""
        self.platforms[platform.id] = platform

    def remove_platform(self, platform_id: str) -> None:
        """Remove a platform from the scenario."""
        if platform_id in self.platforms:
            del self.platforms[platform_id]

    def add_initial_signal(self, signal: Signal) -> None:
        """Add a signal definition to the scenario setup."""
        self.initial_signals.append(signal)


@dataclass
class PulsedWaveform(Waveform):
    """Waveform with pulsed characteristics for high-power applications."""
    pulse_width: float = 1e-6  # Pulse width in seconds
    pulse_repetition_interval: float = 1e-3  # Time between pulses in seconds
    duty_cycle: float = 0.1  # Duty cycle (0-1)
    rise_time: float = 1e-9  # Rise time in seconds
    fall_time: float = 1e-9  # Fall time in seconds
    pulse_shape: str = "rectangular"  # Shape of the pulse (rectangular, gaussian, etc.)
    pulse_shape_params: Dict[str, Any] = field(default_factory=dict)  # Additional shape parameters

@dataclass
class HPMWaveform(PulsedWaveform):
    """High-Power Microwave waveform with specialized parameters."""
    peak_power: float = 1e6  # Peak power in Watts
    energy_per_pulse: float = 1.0  # Energy per pulse in Joules
    polarization: str = "linear"  # Polarization type (linear, circular, etc.)
    beam_width: float = 1.0  # Beam width in degrees
    is_directed: bool = True  # Whether the waveform is directed or broadcast
    target_effects: List[str] = field(default_factory=lambda: ["upset"])  # Intended effects
    
    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        # Calculate duty cycle if not explicitly provided
        if self.duty_cycle == 0.1:  # Default value
            self.duty_cycle = self.pulse_width / self.pulse_repetition_interval
        
        # Calculate average power
        self.average_power = self.peak_power * self.duty_cycle
        
        # Calculate energy per pulse if not provided
        if self.energy_per_pulse == 1.0:  # Default value
            self.energy_per_pulse = self.peak_power * self.pulse_width