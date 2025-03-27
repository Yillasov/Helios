"""Core interfaces and abstract base classes for the Helios RF simulation system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol, Tuple, Set, Union
import numpy as np

from helios.core.data_structures import (
    Scenario, Platform, Signal, Waveform, Position, Velocity, EnvironmentParameters # Added Velocity
)


class IWaveformGenerator(ABC):
    """Interface for waveform generation."""

    @abstractmethod
    def generate(self, waveform: 'Waveform', sampling_rate: float, duration: float) -> np.ndarray:
        """Generate time-domain samples for a waveform.

        Args:
            waveform: The waveform definition.
            sampling_rate: The sampling rate in Hz.
            duration: The duration of the signal in seconds.

        Returns:
            Complex or real time-domain samples.
        """
        pass

    @abstractmethod
    def apply_modulation(self,
                         carrier_samples: np.ndarray,
                         waveform: 'Waveform',
                         sampling_rate: float) -> np.ndarray: # Added sampling_rate
        """Apply modulation to a base carrier signal.

        Args:
            carrier_samples: Baseband or IF carrier samples.
            waveform: Waveform definition containing modulation type and parameters.
            sampling_rate: The sampling rate in Hz.

        Returns:
            Modulated signal samples.
        """
        pass


class IPropagationModel(ABC):
    """Interface for RF propagation models."""
    
    @abstractmethod
    def calculate_path_loss(self, 
                           tx_position: Position, 
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """Calculate path loss between transmitter and receiver.
        
        Args:
            tx_position: Transmitter position
            rx_position: Receiver position
            frequency: Signal frequency in Hz
            environment: Environmental parameters
            
        Returns:
            Path loss in dB
        """
        pass
    
    @abstractmethod
    def apply_propagation_effects(self, 
                                 signal: Signal, # Contains tx_pos, tx_vel, freq, tx_power
                                 rx_platform: Platform, # Provides rx_pos and rx_vel
                                 environment: EnvironmentParameters) -> Signal:
        """Apply propagation effects (path loss, delay, Doppler) to a signal.
        
        Args:
            signal: Original transmitted signal, including source velocity.
            rx_platform: The receiving platform, providing position and velocity.
            environment: Environmental parameters.
        
        Returns:
            Modified signal with propagation effects applied (new power, delay, doppler).
        """
        pass


class IEnvironmentModel(ABC):
    """Interface for environment modeling components."""
    
    @abstractmethod
    def get_terrain_height(self, x: float, y: float) -> float:
        """Get terrain height at a specific location.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            
        Returns:
            Terrain height in meters
        """
        pass
    
    @abstractmethod
    def get_atmospheric_conditions(self, position: Position) -> Dict[str, float]:
        """Get atmospheric conditions at a specific position.
        
        Args:
            position: 3D position
            
        Returns:
            Dictionary of atmospheric parameters (temperature, pressure, etc.)
        """
        pass
    
    @abstractmethod
    def calculate_noise(self, 
                       position: Position, 
                       frequency: float, 
                       bandwidth: float) -> float:
        """Calculate environmental noise at a specific position and frequency.
        
        Args:
            position: 3D position
            frequency: Center frequency in Hz
            bandwidth: Bandwidth in Hz
            
        Returns:
            Noise power in dBm
        """
        pass


class ISignalProcessor(ABC):
    """Interface for signal processing components."""
    
    @abstractmethod
    def detect_signals(self, 
                      received_samples: np.ndarray, 
                      sampling_rate: float,
                      noise_floor: float) -> List[Dict[str, Any]]:
        """Detect signals in received samples.
        
        Args:
            received_samples: Complex samples of received signal
            sampling_rate: Sampling rate in Hz
            noise_floor: Noise floor in dBm
            
        Returns:
            List of detected signal parameters
        """
        pass
    
    @abstractmethod
    def demodulate(self, 
                  samples: np.ndarray, 
                  waveform: Waveform, 
                  sampling_rate: float) -> np.ndarray:
        """Demodulate received signal samples.
        
        Args:
            samples: Received signal samples
            waveform: Waveform definition with modulation parameters
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Demodulated data samples
        """
        pass


class ISimulationEngine(ABC):
    """Interface for simulation engine components."""
    
    @abstractmethod
    def initialize(self, scenario: Scenario) -> None:
        """Initialize the simulation with a scenario.
        
        Args:
            scenario: The scenario to simulate
        """
        pass
    
    @abstractmethod
    def run(self, duration: Optional[float] = None) -> None:
        """Run the simulation for a specified duration.
        
        Args:
            duration: Duration to run in seconds, or None to use scenario duration
        """
        pass
    
    @abstractmethod
    def step(self, time_step: float) -> None:
        """Advance the simulation by a single time step.
        
        Args:
            time_step: Time step in seconds
        """
        pass
    
    @abstractmethod
    def schedule_event(self, 
                      time: float, 
                      event_type: str, 
                      data: Any = None, 
                      callback: Any = None) -> None:
        """Schedule an event to occur at a specific time.
        
        Args:
            time: Simulation time for the event
            event_type: Type of event
            data: Event data
            callback: Function to call when event occurs
        """
        pass
    
    @property
    @abstractmethod
    def current_time(self) -> float:
        """Get the current simulation time.
        
        Returns:
            Current time in seconds
        """
        pass


class IPlatformController(ABC):
    """Interface for platform control components."""
    
    @abstractmethod
    def update_position(self, platform: Platform, time_delta: float) -> None:
        """Update platform position based on current state.
        
        Args:
            platform: The platform to update
            time_delta: Time step in seconds
        """
        pass
    
    @abstractmethod
    def set_waypoint(self, platform: Platform, position: Position, speed: float) -> None:
        """Set a waypoint for platform movement.
        
        Args:
            platform: The platform to control
            position: Target position
            speed: Movement speed in m/s
        """
        pass
    
    @abstractmethod
    def execute_maneuver(self, platform: Platform, maneuver_type: str, parameters: Dict[str, Any]) -> None:
        """Execute a predefined maneuver.
        
        Args:
            platform: The platform to control
            maneuver_type: Type of maneuver (e.g., "orbit", "climb", "descend")
            parameters: Maneuver-specific parameters
        """
        pass


class IDataRecorder(ABC):
    """Interface for simulation data recording components."""
    
    @abstractmethod
    def record_platform_state(self, time: float, platform: Platform) -> None:
        """Record platform state at a specific time.
        
        Args:
            time: Simulation time
            platform: Platform to record
        """
        pass
    
    @abstractmethod
    def record_signal(self, time: float, signal: Signal, receiver_id: str) -> None:
        """Record signal reception at a specific time.
        
        Args:
            time: Simulation time
            signal: Received signal
            receiver_id: ID of the receiving system
        """
        pass
    
    @abstractmethod
    def save_results(self, filename: str) -> None:
        """Save recorded results to a file.
        
        Args:
            filename: Output filename
        """
        pass