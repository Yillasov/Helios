"""Machine Learning integration for cognitive algorithms."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import os
import requests
import time 
# Import 'field' from dataclasses
from dataclasses import dataclass, field
import random # Import random for exploration

from helios.cognitive.data_structures import EnvironmentState, SpectrumBand
from helios.core.data_structures import CognitiveWaveform, AdaptationGoal
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML model integration."""
    model_type: str  # "local", "api", "rl", "custom" # Added "rl"
    endpoint_url: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    # Use field for default factory, now that it's imported
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    # RL specific params (optional)
    rl_params: Optional[Dict[str, Any]] = None


class MLIntegration:
    """Provides integration with machine learning frameworks via APIs and RL."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ML integration with optional config file."""
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, MLModelConfig] = {}

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Initialize API session for reuse
        self.session = requests.Session()

        # Feature history for training/evaluation
        self.feature_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000

        # --- RL Specific State ---
        # Simple Q-table like structure: state -> action -> value
        # State representation needs careful design in a real system
        self.q_values: Dict[str, Dict[str, float]] = {}
        self.rl_learning_rate = 0.1
        self.rl_discount_factor = 0.9
        self.rl_exploration_rate = 0.2 # Epsilon for epsilon-greedy

    def _load_config(self, config_path: str) -> None:
        """Load ML model configurations from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for model_name, model_config in config_data.get('models', {}).items():
                self.model_configs[model_name] = MLModelConfig(**model_config)
                logger.info(f"Loaded configuration for ML model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")

    def predict_adaptation(self,
                          model_name: str,
                          env_state: EnvironmentState,
                          waveform: CognitiveWaveform) -> Dict[str, Any]:
        """
        Get adaptation predictions from the specified ML model.
        """
        if model_name not in self.model_configs:
            logger.warning(f"Unknown ML model: {model_name}")
            return {}

        config = self.model_configs[model_name]

        # Prepare input features (might be used differently by RL)
        features = self._extract_features(env_state, waveform)

        # Get prediction based on model type
        if config.model_type == "api":
            return self._predict_api(config, features)
        elif config.model_type == "local":
            return self._predict_local(config, features)
        elif config.model_type == "rl": # Handle RL prediction
            return self._predict_rl(config, features, env_state, waveform)
        else:
            logger.warning(f"Unsupported model type: {config.model_type}")
            return {}

    def _extract_features(self,
                         env_state: EnvironmentState,
                         waveform: CognitiveWaveform) -> Dict[str, Any]:
        """Extract relevant features for ML model input."""
        features = {
            "timestamp": env_state.timestamp,
            "center_frequency": waveform.center_frequency,
            "bandwidth": waveform.bandwidth,
            "amplitude": waveform.amplitude,
            "modulation_type": str(waveform.modulation_type),
            "adaptation_goals": [str(goal) for goal in waveform.adaptation_goals],
        }
        
        # Add spectrum occupancy features
        if env_state.spectrum_occupancy:
            occupied_bands = []
            for band_id, occupancy in env_state.spectrum_occupancy.items():
                occupied_bands.append({
                    "start_freq": occupancy.band.start_freq,
                    "end_freq": occupancy.band.end_freq,
                    "power_level": occupancy.power_level,
                    "is_occupied": occupancy.is_occupied,
                    "snr": occupancy.snr
                })
            features["spectrum_bands"] = occupied_bands
        
        return features

    def _predict_api(self, config: MLModelConfig, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from remote API endpoint."""
        if not config.endpoint_url:
            logger.error("No endpoint URL configured for API model")
            return {}
        
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        
        try:
            response = self.session.post(
                config.endpoint_url,
                json={"features": features},
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("adaptations", {})
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error calling ML API: {e}")
            return {}

    def _predict_local(self, config: MLModelConfig, features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from locally loaded model."""
        # This is a placeholder for local model integration
        # In a real implementation, this would use frameworks like scikit-learn, TensorFlow, etc.
        logger.info("Local ML model prediction not yet implemented")
        return {}

    def _predict_rl(self,
                    config: MLModelConfig,
                    features: Dict[str, Any],
                    env_state: EnvironmentState,
                    waveform: CognitiveWaveform) -> Dict[str, Any]:
        """Reinforcement Learning based prediction."""
        logger.info(f"Predicting adaptation using RL model: {config.model_path}")
    
        # --- State Representation ---
        # Get current frequency band status
        current_freq_band = None
        for band_id, occupancy in env_state.spectrum_occupancy.items():
            if (occupancy.band.start_freq <= waveform.center_frequency <= 
                occupancy.band.end_freq):
                current_freq_band = occupancy
                break
        
        jamming_status = "jammed" if current_freq_band and current_freq_band.is_occupied else "clear"
        snr_level = "low" if current_freq_band and current_freq_band.snr < 10 else "high"
        
        # Fix: Use the enum name instead of trying to access goal_type
        goal_name = waveform.adaptation_goals[0].name if waveform.adaptation_goals else 'none'
        
        # More comprehensive state representation
        state_key = f"jamming_{jamming_status}_snr_{snr_level}_goal_{goal_name}"
    
        # --- Action Space ---
        # Define possible actions with constraints
        possible_actions = {
            "freq_up": {"center_frequency": min(waveform.center_frequency + waveform.bandwidth, 
                                            waveform.adaptation_constraints.get('max_frequency', 6e9))},
            "freq_down": {"center_frequency": max(waveform.center_frequency - waveform.bandwidth, 
                                            waveform.adaptation_constraints.get('min_frequency', 30e6))},
            "power_up": {"amplitude": min(waveform.amplitude * 1.2, 
                                        waveform.adaptation_constraints.get('max_amplitude', 1.0))},
            "power_down": {"amplitude": max(waveform.amplitude * 0.8, 
                                        waveform.adaptation_constraints.get('min_amplitude', 0.1))},
            "bandwidth_up": {"bandwidth": min(waveform.bandwidth * 1.2, 
                                        waveform.adaptation_constraints.get('max_bandwidth', 20e6))},
            "bandwidth_down": {"bandwidth": max(waveform.bandwidth * 0.8, 
                                            waveform.adaptation_constraints.get('min_bandwidth', 1e6))},
            "no_change": {}
        }
        
        action_keys = list(possible_actions.keys())
    
        # Initialize Q-values for state if not seen
        if state_key not in self.q_values:
            self.q_values[state_key] = {action: 0.0 for action in action_keys}
    
        # --- Epsilon-Greedy Action Selection ---
        if random.random() < self.rl_exploration_rate:
            # Explore: Choose a random action
            action_key = random.choice(action_keys)
            logger.debug(f"RL Explore: Chose random action '{action_key}'")
        else:
            # Exploit: Choose the best known action for this state
            q_s = self.q_values[state_key]
            action_key = max(q_s.keys(), key=lambda k: q_s[k])
            logger.debug(f"RL Exploit: Chose best action '{action_key}' with value {q_s[action_key]:.2f}")
    
        chosen_action_params = possible_actions[action_key]
    
        # Store chosen action for learning update
        # We need a way to link this prediction to the subsequent reward
        self._last_rl_prediction = {
            "state": state_key,
            "action": action_key
        }
    
        # Return the parameter changes for the chosen action
        return chosen_action_params


    def record_adaptation_result(self,
                               features: Dict[str, Any],
                               adaptations: Dict[str, Any],
                               performance_metrics: Dict[str, float]) -> None:
        """
        Record adaptation results for future training and RL updates.
        """
        # Validate inputs
        if not isinstance(performance_metrics, dict):
            logger.error(f"Invalid performance metrics: {performance_metrics}")
            return
            
        # Create record with proper timestamp
        record = {
            "features": features,
            "adaptations": adaptations,
            "performance": performance_metrics,
            "timestamp": features.get("timestamp", time.time())
        }
    
        # --- Update RL Q-values ---
        if hasattr(self, '_last_rl_prediction') and self._last_rl_prediction:
            state = self._last_rl_prediction["state"]
            action = self._last_rl_prediction["action"]
    
            # Calculate reward based on multiple performance metrics
            reward = 0.0
            if 'snr' in performance_metrics:
                reward += performance_metrics['snr'] * 0.5  # Weight SNR heavily
            if 'throughput' in performance_metrics:
                reward += performance_metrics['throughput'] * 0.3  # Weight throughput
            if 'interference_db' in performance_metrics:
                reward -= performance_metrics['interference_db'] * 0.2  # Penalize interference
                
            # Apply Q-learning update with proper discount factor
            if state in self.q_values and action in self.q_values[state]:
                old_value = self.q_values[state][action]
                # Full Q-learning update with discount factor
                # We don't have next state here, so simplify
                new_value = old_value + self.rl_learning_rate * (reward - old_value)
                self.q_values[state][action] = new_value
                
                logger.debug(f"RL Update: State='{state}', Action='{action}', Reward={reward:.2f}, Q={new_value:.2f}")
    
            # Clear last prediction
            self._last_rl_prediction = None
    
        # Store feature history with size limit
        self.feature_history.append(record)
        if len(self.feature_history) > self.max_history_size:
            self.feature_history = self.feature_history[-self.max_history_size:]