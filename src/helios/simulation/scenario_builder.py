# New file to demonstrate building a scenario from config

import logging
from typing import Dict, Any

from helios.core.data_structures import (
    Scenario, Platform, Position, Velocity, Orientation, System,
    EnvironmentParameters, Waveform, ModulationType, Signal
)
from helios.utils.logger import get_logger
from helios.utils.config_loader import load_config

logger = get_logger(__name__)

def _parse_position(pos_data: Dict[str, float]) -> Position:
    return Position(x=pos_data.get('x', 0.0), y=pos_data.get('y', 0.0), z=pos_data.get('z', 0.0))

def _parse_velocity(vel_data: Dict[str, float]) -> Velocity:
    return Velocity(x=vel_data.get('x', 0.0), y=vel_data.get('y', 0.0), z=vel_data.get('z', 0.0))

def _parse_orientation(orient_data: Dict[str, float]) -> Orientation:
    return Orientation(roll=orient_data.get('roll', 0.0), pitch=orient_data.get('pitch', 0.0), yaw=orient_data.get('yaw', 0.0))

def build_scenario_from_config(config_path: str) -> Scenario:
    """
    Builds a Scenario object from a YAML configuration file.

    Args:
        config_path: Path to the scenario configuration YAML file.

    Returns:
        A configured Scenario object.

    Raises:
        ValueError: If the configuration is missing required sections or has invalid data.
    """
    config = load_config(config_path)

    sim_config = config.get('simulation', {})
    env_config = config.get('environment', {})
    platform_configs = config.get('platforms', [])
    waveform_configs = config.get('waveforms', [])
    # initial_signals_config = config.get('initial_signals', []) # Placeholder for later

    # Create Environment Parameters
    env_params = EnvironmentParameters(
        temperature=env_config.get('temperature', 290.0),
        noise_floor_density=env_config.get('noise_floor_density', -174.0)
    )
    logger.debug(f"Created environment parameters: {env_params}")

    # Create Scenario object
    scenario = Scenario(
        name=sim_config.get('scenario_name', 'Unnamed Scenario'),
        description=sim_config.get('description', ''),
        start_time=sim_config.get('start_time', 0.0),
        duration=sim_config.get('duration', 3600.0),
        environment=env_params
    )
    logger.info(f"Created scenario '{scenario.name}' (ID: {scenario.id})")

    # Load Waveforms (store them temporarily for lookup)
    waveforms: Dict[str, Waveform] = {}
    for wf_conf in waveform_configs:
        wf_id = wf_conf.get('id')
        if not wf_id:
            logger.warning("Skipping waveform config with missing 'id'")
            continue
        try:
            mod_type_str = wf_conf.get('modulation_type', 'NONE').upper()
            mod_type = ModulationType[mod_type_str]
        except KeyError:
            logger.warning(f"Invalid modulation type '{wf_conf.get('modulation_type')}' for waveform {wf_id}. Defaulting to NONE.")
            mod_type = ModulationType.NONE

        waveform = Waveform(
            id=wf_id,
            center_frequency=wf_conf.get('center_frequency', 0.0),
            bandwidth=wf_conf.get('bandwidth', 0.0),
            amplitude=wf_conf.get('amplitude', 1.0),
            modulation_type=mod_type,
            modulation_params=wf_conf.get('modulation_params', {}),
            duration=wf_conf.get('duration') # Optional duration
        )
        waveforms[wf_id] = waveform
        logger.debug(f"Loaded waveform {wf_id}: {waveform}")


    # Create Platforms and their Systems
    for plat_conf in platform_configs:
        platform_id = plat_conf.get('id')
        if not platform_id:
            logger.warning("Skipping platform config with missing 'id'")
            continue

        platform = Platform(
            id=platform_id,
            name=plat_conf.get('name', 'Unnamed Platform'),
            position=_parse_position(plat_conf.get('position', {})),
            velocity=_parse_velocity(plat_conf.get('velocity', {})),
            orientation=_parse_orientation(plat_conf.get('orientation', {})),
            rcs=plat_conf.get('rcs') # Optional RCS
        )
        logger.debug(f"Created platform '{platform.name}' ({platform.id})")

        # Add Systems to Platform
        system_configs = plat_conf.get('systems', [])
        for sys_conf in system_configs:
            system_id = sys_conf.get('id')
            if not system_id:
                logger.warning(f"Skipping system config on platform {platform_id} with missing 'id'")
                continue
            system = System(
                id=system_id,
                name=sys_conf.get('name', 'Unnamed System'),
                system_type=sys_conf.get('system_type', 'unknown'),
                parameters=sys_conf.get('parameters', {}) # Store raw params dict
            )
            platform.add_system(system)
            logger.debug(f"  Added system '{system.name}' ({system.id}) to platform {platform.id}")

        scenario.add_platform(platform)

    # TODO: Load initial signals (linking waveforms and source systems)
    # initial_signals_config = config.get('initial_signals', [])
    # for sig_conf in initial_signals_config:
    #    ... create Signal objects and add using scenario.add_initial_signal ...

    logger.info(f"Scenario built successfully from {config_path}")
    return scenario

# Example of how to use the builder:
# if __name__ == "__main__":
#     try:
#         config_file = "/Users/yessine/Helios/config/default_sim_config.yaml"
#         # Ensure the example config exists for standalone run
#         if not os.path.exists(config_file):
#             raise FileNotFoundError(f"Run the config creation step first (e.g., manually or via prior script step)")
#
#         # Setu^p basic logging for the builder example
#         import sys
#         from helios.utils.logger import setup_logging
#         setup_logging(level=logging.DEBUG, stream=sys.stdout)
#
#         my_scenario = build_scenario_from_config(config_file)
#         print("\n--- Built Scenario ---")
#         print(f"Name: {my_scenario.name}")
#         print(f"Duration: {my_scenario.duration}")
#         print(f"Environment Temp: {my_scenario.environment.temperature} K")
#         print(f"Platforms ({len(my_scenario.platforms)}):")
#         for p_id, p in my_scenario.platforms.items():
#             print(f"  - {p.name} ({p_id}) at {p.position}")
#             print(f"    Systems ({len(p.equipped_systems)}): {[s.name for s in p.equipped_systems.values()]}")
#
#     except Exception as e:
#         print(f"Error building scenario: {e}")