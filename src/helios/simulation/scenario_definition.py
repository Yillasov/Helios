"""Tools for programmatically defining simulation scenarios."""

from typing import Dict, Any, Optional, List, Union

from helios.core.data_structures import (
    Scenario, Platform, Position, Velocity, Orientation, System,
    EnvironmentParameters, Waveform, ModulationType, Signal
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

def create_static_platform(
    platform_id: str,
    name: str,
    position: Position,
    systems: Optional[List[System]] = None,
    orientation: Optional[Orientation] = None,
    rcs: Optional[Union[float, Dict[str, float]]] = None
) -> Platform:
    """
    Creates a Platform object representing a static entity.

    Args:
        platform_id: Unique identifier for the platform.
        name: Display name for the platform.
        position: Initial position (x, y, z) of the platform.
        systems: Optional list of systems equipped on the platform.
        orientation: Optional initial orientation (roll, pitch, yaw).
        rcs: Optional Radar Cross Section.

    Returns:
        A configured Platform object with zero velocity.
    """
    platform = Platform(
        id=platform_id,
        name=name,
        position=position,
        velocity=Velocity(x=0.0, y=0.0, z=0.0), # Static platform
        orientation=orientation or Orientation(),
        rcs=rcs
    )
    if systems:
        for system in systems:
            platform.add_system(system)
    logger.debug(f"Created static platform: {platform.id} ({platform.name}) at {platform.position}")
    return platform

def create_basic_signal(
    signal_id: str,
    source_platform_id: str,
    frequency: float,
    power: float, # Typically in dBm
    bandwidth: float = 0.0,
    emission_time: float = 0.0,
    duration: Optional[float] = None,
    modulation: ModulationType = ModulationType.NONE,
    mod_params: Optional[Dict[str, Any]] = None
) -> Signal:
    """
    Creates a basic Signal object with a simple waveform.

    Args:
        signal_id: Unique identifier for the signal instance.
        source_platform_id: ID of the platform emitting the signal.
        frequency: Center frequency in Hz.
        power: Transmit power in dBm.
        bandwidth: Signal bandwidth in Hz (default 0 for CW).
        emission_time: Simulation time when the signal starts emitting.
        duration: Optional duration of the signal emission in seconds.
        modulation: Type of modulation (default NONE).
        mod_params: Optional dictionary of modulation parameters.

    Returns:
        A configured Signal object.
    """
    # Create an embedded Waveform for simplicity here
    waveform = Waveform(
        id=f"wf_{signal_id}", # Generate a unique waveform ID
        center_frequency=frequency,
        bandwidth=bandwidth,
        modulation_type=modulation,
        modulation_params=mod_params or {},
        duration=duration # Duration correctly passed to Waveform
    )
    signal = Signal(
        id=signal_id,
        source_id=source_platform_id,
        waveform=waveform,
        power=power,
        emission_time=emission_time,
        # duration=duration # Remove this line - Signal does not take duration directly
        # tx_position and tx_velocity will be set by the engine at emission time
    )
    logger.debug(f"Created basic signal: {signal.id} from {signal.source_id} at {signal.emission_time}s")
    return signal

# --- Example Usage ---
if __name__ == "__main__":
    # Define environment
    env = EnvironmentParameters(temperature=290, noise_floor_density=-174)

    # Create platforms
    platform1 = create_static_platform(
        platform_id="ground_tx",
        name="Ground Transmitter",
        position=Position(x=0, y=0, z=10)
    )
    platform2 = create_static_platform(
        platform_id="air_rx",
        name="Airborne Receiver",
        position=Position(x=10000, y=5000, z=8000)
    )

    # Create signals
    signal1 = create_basic_signal(
        signal_id="sig_pulse_1",
        source_platform_id=platform1.id,
        frequency=1.2e9, # 1.2 GHz
        power=30.0, # 30 dBm (1 Watt)
        bandwidth=1e6, # 1 MHz
        emission_time=1.0,
        duration=0.01 # 10 ms pulse
    )
    signal2 = create_basic_signal(
        signal_id="sig_cw_1",
        source_platform_id=platform1.id,
        frequency=1.205e9, # 1.205 GHz
        power=20.0, # 20 dBm
        emission_time=5.0
        # No duration = continuous wave after emission_time
    )

    # Create scenario
    scenario = Scenario(
        name="Basic Static Scenario",
        description="A simple scenario with one static transmitter and one static receiver.",
        start_time=0.0,
        duration=20.0, # Simulate for 20 seconds
        environment=env
    )

    # Add elements to scenario
    scenario.add_platform(platform1)
    scenario.add_platform(platform2)
    scenario.add_initial_signal(signal1)
    scenario.add_initial_signal(signal2)

    logger.info(f"Scenario '{scenario.name}' created programmatically.")
    logger.info(f"  Platforms: {len(scenario.platforms)}")
    logger.info(f"  Initial Signals: {len(scenario.initial_signals)}")

    # This scenario object can now be passed to the SimulationEngine
    # from helios.simulation.engine import SimulationEngine
    # engine = SimulationEngine(scenario)
    # engine.run()