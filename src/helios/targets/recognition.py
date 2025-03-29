"""Target recognition algorithms for RF signatures."""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
import time

from helios.core.data_structures import Signal, Position
from helios.targets.rf_signature_db import RFSignatureDatabase, RFSignature
from helios.utils.logger import get_logger
# Import the waveform subclasses to properly check types
from helios.core.data_structures import PulsedWaveform
from helios.waveforms.advanced_waveforms import FrequencyHoppingWaveform

logger = get_logger(__name__)

class TargetRecognitionAlgorithm:
    """Base class for target recognition algorithms."""
    
    def __init__(self, signature_db: RFSignatureDatabase):
        """Initialize with a signature database."""
        self.signature_db = signature_db
        
    def recognize(self, signal: Signal) -> List[Tuple[RFSignature, float]]:
        """
        Recognize a target from its RF signal.
        
        Args:
            signal: Detected RF signal
            
        Returns:
            List of (signature, confidence) tuples, sorted by confidence
        """
        # Basic implementation uses the database's matching function
        return self.signature_db.identify_signal(signal)

class FeatureBasedRecognition(TargetRecognitionAlgorithm):
    """Recognition algorithm based on feature extraction and matching."""
    
    def extract_features(self, signal: Signal) -> Dict[str, Any]:
        """Extract features from a signal for recognition."""
        features = {
            "center_frequency": signal.waveform.center_frequency,
            "bandwidth": signal.waveform.bandwidth,
            "modulation_type": signal.waveform.modulation_type,
            "power": signal.power
        }
        
        # Extract pulse width if it's a PulsedWaveform
        if isinstance(signal.waveform, PulsedWaveform):
            features["pulse_width"] = signal.waveform.pulse_width
            
        # Extract hop rate if it's a FrequencyHoppingWaveform
        if isinstance(signal.waveform, FrequencyHoppingWaveform):
            features["frequency_hopping"] = True
            features["hop_rate"] = signal.waveform.hop_interval  # hop_interval is the attribute in FrequencyHoppingWaveform
            
        return features
    
    def recognize(self, signal: Signal) -> List[Tuple[RFSignature, float]]:
        """Enhanced recognition using feature extraction."""
        features = self.extract_features(signal)
        matches = []
        
        for signature in self.signature_db.signatures.values():
            score = self._calculate_feature_match(features, signature)
            if score > 0.6:  # Minimum threshold
                matches.append((signature, score))
                
        # Sort by score, highest first
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _calculate_feature_match(self, features: Dict[str, Any], signature: RFSignature) -> float:
        """Calculate match score based on feature comparison."""
        score = 0.0
        total_weight = 0.0
        
        # Frequency match (highest weight)
        weight = 3.0
        total_weight += weight
        if abs(features["center_frequency"] - signature.features.center_frequency) < signature.features.bandwidth:
            freq_match = 1.0 - abs(features["center_frequency"] - signature.features.center_frequency) / signature.features.bandwidth
            score += weight * max(0.0, freq_match)
        
        # Bandwidth match
        weight = 2.0
        total_weight += weight
        if features["bandwidth"] > 0 and signature.features.bandwidth > 0:
            bw_ratio = min(features["bandwidth"], signature.features.bandwidth) / max(features["bandwidth"], signature.features.bandwidth)
            score += weight * bw_ratio
        
        # Modulation match
        weight = 2.0
        total_weight += weight
        if features["modulation_type"] == signature.features.modulation_type:
            score += weight
        
        # Additional feature matches
        for key in ["pulse_width", "hop_rate"]:
            if key in features and getattr(signature.features, key):
                weight = 1.0
                total_weight += weight
                feature_val = features[key]
                signature_val = getattr(signature.features, key)
                
                # Calculate similarity based on relative difference
                if feature_val > 0 and signature_val > 0:
                    similarity = min(feature_val, signature_val) / max(feature_val, signature_val)
                    score += weight * similarity
        
        # Normalize score
        return score / total_weight if total_weight > 0 else 0.0