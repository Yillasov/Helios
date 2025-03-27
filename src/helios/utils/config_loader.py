import yaml
import os
from typing import Dict, Any
from .logger import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration parameters from a YAML file.

    Args:
        config_path: The absolute or relative path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        Exception: For other potential file reading errors.
    """
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            if config is None: # Handle empty file case
                logger.warning(f"Configuration file is empty: {config_path}")
                return {}
            logger.debug(f"Successfully loaded configuration: {config}")
            return config
    except yaml.YAMLError as exc:
        error_msg = f"Error parsing YAML file {config_path}: {exc}"
        logger.error(error_msg)
        raise # Re-raise the YAML error
    except Exception as exc:
        error_msg = f"Error reading configuration file {config_path}: {exc}"
        logger.error(error_msg, exc_info=True)
        raise # Re-raise other exceptions

# Example usage (can be placed in a main script or test)
# if __name__ == "__main__":
#     # Assuming a config file exists at ../../config/default_sim_config.yaml
#     try:
#         config_dir = os.path.join(os.path.dirname(__file__), '../../config')
#         default_config_path = os.path.join(config_dir, 'default_sim_config.yaml')
#         # Create a dummy config if it doesn't exist for testing
#         if not os.path.exists(default_config_path):
#             os.makedirs(config_dir, exist_ok=True)
#             with open(default_config_path, 'w') as f:
#                 yaml.dump({'simulation': {'duration': 10.0, 'time_step': 0.1}, 'logging': {'level': 'INFO'}}, f)
#
#         config_data = load_config(default_config_path)
#         print("Config loaded successfully:")
#         print(config_data)
#         sim_duration = config_data.get('simulation', {}).get('duration', 1.0)
#         print(f"Simulation duration from config: {sim_duration}")
#
#     except Exception as e:
#         print(f"Failed to load configuration: {e}")