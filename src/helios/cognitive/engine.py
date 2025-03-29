"""Cognitive engine for adaptive waveform generation."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import random
import os

from helios.cognitive.data_structures import EnvironmentState, SpectrumBand, SpectrumOccupancy
from helios.core.data_structures import Waveform, CognitiveWaveform, AdaptationGoal
from helios.utils.logger import get_logger
# Import the new ML integration
from helios.cognitive.ml_integration import MLIntegration

logger = get_logger(__name__)

class CognitiveEngine:
    """Engine for making cognitive adaptation decisions based on environmental feedback."""
    
    def __init__(self, ml_config_path: Optional[str] = None):
        """Initialize the cognitive engine."""
        self.current_state: Optional[EnvironmentState] = None
        self.adaptation_history: List[Dict[str, Any]] = []
        self.learning_rate: float = 0.1  # For future reinforcement learning
        
        # Initialize ML integration with optional config
        config_path = ml_config_path or os.environ.get("HELIOS_ML_CONFIG")
        self.ml_integration = MLIntegration(config_path)
        
        # Track performance metrics for ML feedback
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # ML model selection strategy (can be rule-based, random, or performance-based)
        self.ml_strategy = "hybrid"  # Options: "rule", "ml", "hybrid"
    
    def update_environment_state(self, state: EnvironmentState) -> None:
        """Update the current environment state."""
        self.current_state = state
        logger.debug(f"Cognitive engine updated with environment state at t={state.timestamp}")
    
    def decide_adaptation(self, waveform: CognitiveWaveform) -> Dict[str, Any]:
        """
        Decide how to adapt a waveform based on current environment state.
        Now with ML integration for enhanced decision making.
        
        Args:
            waveform: The cognitive waveform to adapt
            
        Returns:
            Dictionary of parameter changes to apply
        """
        if not self.current_state:
            logger.warning("No environment state available for adaptation decision")
            return {}
        
        adaptations = {}
        
        # Extract features for ML (regardless of whether we use ML now)
        features = self.ml_integration._extract_features(self.current_state, waveform)
        
        # Determine whether to use ML or rule-based approach
        use_ml = self._should_use_ml(waveform)
        
        if use_ml:
            # Get ML-based adaptations
            ml_model = self._select_best_ml_model(waveform)
            ml_adaptations = self.ml_integration.predict_adaptation(ml_model, self.current_state, waveform)
            
            if ml_adaptations:
                logger.info(f"Using ML model '{ml_model}' for adaptation")
                adaptations.update(ml_adaptations)
            else:
                # Fall back to rule-based if ML fails
                logger.info("ML adaptation failed, falling back to rule-based")
                use_ml = False
        
        # If not using ML or ML failed, use rule-based approach
        if not use_ml:
            # Process each adaptation goal with rule-based methods
            for goal in waveform.adaptation_goals:
                if goal == AdaptationGoal.MINIMIZE_INTERFERENCE:
                    freq_adaptation = self._adapt_for_interference_avoidance(waveform)
                    if freq_adaptation:
                        adaptations.update(freq_adaptation)
                
                elif goal == AdaptationGoal.MAXIMIZE_SNR:
                    snr_adaptation = self._adapt_for_snr_maximization(waveform)
                    if snr_adaptation:
                        adaptations.update(snr_adaptation)
        
        # Record adaptation decision
        if adaptations:
            adaptation_record = {
                "timestamp": self.current_state.timestamp,
                "waveform_id": waveform.id,
                "adaptations": adaptations,
                "method": "ml" if use_ml else "rule-based"
            }
            self.adaptation_history.append(adaptation_record)
        
        return adaptations
    
    def _should_use_ml(self, waveform: CognitiveWaveform) -> bool:
        """Determine whether to use ML for this adaptation decision."""
        if self.ml_strategy == "ml":
            return True
        elif self.ml_strategy == "rule":
            return False
        else:  # hybrid strategy
            # Use ML more often as we collect more data
            history_size = len(self.adaptation_history)
            ml_probability = min(0.8, 0.2 + (history_size / 1000) * 0.6)
            return random.random() < ml_probability
    
    def _select_best_ml_model(self, waveform: CognitiveWaveform) -> str:
        """Select the best ML model based on past performance."""
        # Simple implementation - in practice would use more sophisticated selection
        available_models = list(self.ml_integration.model_configs.keys())
        if not available_models:
            return "default"
        
        # For now, just return the first available model
        return available_models[0]
    
    def record_performance_metrics(self, waveform_id: str, metrics: Dict[str, float]) -> None:
        """
        Record performance metrics for a waveform after adaptation.
        This provides feedback for ML models.
        
        Args:
            waveform_id: ID of the waveform
            metrics: Dictionary of performance metrics (SNR, throughput, etc.)
        """
        if not self.current_state:
            return
        
        self.performance_metrics[waveform_id] = {
            "timestamp": self.current_state.timestamp,
            **metrics
        }
        
        # Find the most recent adaptation for this waveform
        for adaptation in reversed(self.adaptation_history):
            if adaptation.get("waveform_id") == waveform_id:
                # Record result for ML training
                features = adaptation.get("features", {})
                adaptations = adaptation.get("adaptations", {})
                
                self.ml_integration.record_adaptation_result(
                    features, adaptations, metrics
                )
                break
    
    def _adapt_for_interference_avoidance(self, waveform: CognitiveWaveform) -> Dict[str, Any]:
        """
        Implement frequency hopping to avoid interference.
        
        Args:
            waveform: The cognitive waveform to adapt
            
        Returns:
            Dictionary with new center_frequency if adaptation needed
        """
        # Check if current frequency band is experiencing interference
        current_band = self.current_state.get_band_occupancy(waveform.center_frequency) if self.current_state else None
        
        # If no information about current band or no interference, no adaptation needed
        if not current_band or not current_band.is_occupied:
            return {}
        
        # Find clear bands that meet bandwidth requirements
        required_bw = waveform.bandwidth
        clear_bands = self.current_state.find_clear_bands(required_bw) if self.current_state else []
        
        # If no clear bands available, try to find the least occupied band
        if not clear_bands:
            logger.info("No clear bands available, searching for least occupied band")
            least_occupied = self._find_least_occupied_band(required_bw)
            if least_occupied:
                clear_bands = [least_occupied.band]
        
        # If we found suitable bands, select one (randomly for now)
        if clear_bands:
            # Check frequency constraints
            min_freq = waveform.adaptation_constraints.get('min_frequency', 0)
            max_freq = waveform.adaptation_constraints.get('max_frequency', float('inf'))
            
            # Filter bands by constraints
            valid_bands = [band for band in clear_bands 
                          if band.center_freq >= min_freq and band.center_freq <= max_freq]
            
            if valid_bands:
                # For now, simple random selection - could be improved with learning
                selected_band = random.choice(valid_bands)
                logger.info(f"Adapting waveform {waveform.id} frequency from "
                           f"{waveform.center_frequency/1e6:.2f} MHz to "
                           f"{selected_band.center_freq/1e6:.2f} MHz to avoid interference")
                
                return {"center_frequency": selected_band.center_freq}
        
        logger.warning(f"Could not find suitable band for waveform {waveform.id}")
        return {}
    
    def _adapt_for_snr_maximization(self, waveform: CognitiveWaveform) -> Dict[str, Any]:
        """
        Adapt parameters to maximize SNR.
        
        Args:
            waveform: The cognitive waveform to adapt
            
        Returns:
            Dictionary with parameter changes
        """
        # Get current SNR if available
        current_band = self.current_state.get_band_occupancy(waveform.center_frequency) if self.current_state else None
        if not current_band:
            return {}
        
        current_snr = current_band.snr
        target_snr = waveform.adaptation_constraints.get('target_snr', 10.0)
        
        adaptations = {}
        
        # If SNR is below target, try to increase power
        if current_snr < target_snr:
            max_amplitude = waveform.adaptation_constraints.get('max_amplitude', 1.0)
            
            # Simple linear increase based on SNR difference
            snr_diff = target_snr - current_snr
            amplitude_increase = min(snr_diff * 0.05, 0.2)  # Limit increase to 20%
            
            new_amplitude = min(waveform.amplitude * (1 + amplitude_increase), max_amplitude)
            
            if new_amplitude > waveform.amplitude:
                logger.info(f"Increasing amplitude from {waveform.amplitude:.2f} to {new_amplitude:.2f} "
                           f"to improve SNR from {current_snr:.1f} dB to target {target_snr:.1f} dB")
                adaptations["amplitude"] = new_amplitude
        
        return adaptations
    
    def _find_least_occupied_band(self, required_bandwidth: float) -> Optional[SpectrumOccupancy]:
        """Find the band with the lowest interference level that meets bandwidth requirements."""
        if not self.current_state or not self.current_state.spectrum_occupancy:
            return None
        
        suitable_bands = [occ for occ in self.current_state.spectrum_occupancy.values() 
                         if occ.band.bandwidth >= required_bandwidth]
        
        if not suitable_bands:
            return None
        
        # Sort by SNR (lower is better for finding least interference)
        suitable_bands.sort(key=lambda x: x.snr)
        return suitable_bands[0]
