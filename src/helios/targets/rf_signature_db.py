"""Database system for storing and retrieving target RF signatures."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
import numpy as np
import os
import time
from pathlib import Path

from helios.core.data_structures import Signal, Waveform
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class SignatureType(Enum):
    """Types of RF signatures."""
    COMMUNICATION = auto()
    RADAR = auto()
    NAVIGATION = auto()
    ELECTRONIC_WARFARE = auto()
    UNINTENTIONAL_EMISSION = auto()
    UNKNOWN = auto()

@dataclass
class RFSignatureFeatures:
    """Features that characterize an RF signature."""
    # Frequency domain features
    center_frequency: float = 0.0  # Hz
    bandwidth: float = 0.0  # Hz
    frequency_hopping: bool = False
    hop_rate: Optional[float] = None  # Hz
    frequency_range: Optional[Tuple[float, float]] = None  # Hz
    
    # Time domain features
    pulse_width: Optional[float] = None  # seconds
    pulse_repetition_interval: Optional[float] = None  # seconds
    duty_cycle: Optional[float] = None  # percentage
    
    # Modulation features
    modulation_type: Optional[str] = None
    symbol_rate: Optional[float] = None  # symbols/second
    
    # Power features
    typical_power: Optional[float] = None  # dBm
    power_range: Optional[Tuple[float, float]] = None  # dBm
    
    # Additional features
    polarization: Optional[str] = None
    scan_pattern: Optional[str] = None  # For radar
    additional_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RFSignature:
    """RF signature of a target."""
    id: str
    name: str
    signature_type: SignatureType
    features: RFSignatureFeatures
    platform_types: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0
    notes: str = ""
    last_updated: float = field(default_factory=time.time)
    sample_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "signature_type": self.signature_type.name,
            "features": {
                "center_frequency": self.features.center_frequency,
                "bandwidth": self.features.bandwidth,
                "frequency_hopping": self.features.frequency_hopping,
                "hop_rate": self.features.hop_rate,
                "frequency_range": self.features.frequency_range,
                "pulse_width": self.features.pulse_width,
                "pulse_repetition_interval": self.features.pulse_repetition_interval,
                "duty_cycle": self.features.duty_cycle,
                "modulation_type": self.features.modulation_type,
                "symbol_rate": self.features.symbol_rate,
                "typical_power": self.features.typical_power,
                "power_range": self.features.power_range,
                "polarization": self.features.polarization,
                "scan_pattern": self.features.scan_pattern,
                "additional_features": self.features.additional_features
            },
            "platform_types": self.platform_types,
            "confidence": self.confidence,
            "notes": self.notes,
            "last_updated": self.last_updated,
            "sample_signals": self.sample_signals
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RFSignature':
        """Create from dictionary."""
        features = RFSignatureFeatures(
            center_frequency=data["features"]["center_frequency"],
            bandwidth=data["features"]["bandwidth"],
            frequency_hopping=data["features"]["frequency_hopping"],
            hop_rate=data["features"]["hop_rate"],
            frequency_range=data["features"]["frequency_range"],
            pulse_width=data["features"]["pulse_width"],
            pulse_repetition_interval=data["features"]["pulse_repetition_interval"],
            duty_cycle=data["features"]["duty_cycle"],
            modulation_type=data["features"]["modulation_type"],
            symbol_rate=data["features"]["symbol_rate"],
            typical_power=data["features"]["typical_power"],
            power_range=data["features"]["power_range"],
            polarization=data["features"]["polarization"],
            scan_pattern=data["features"]["scan_pattern"],
            additional_features=data["features"]["additional_features"]
        )
        
        return cls(
            id=data["id"],
            name=data["name"],
            signature_type=SignatureType[data["signature_type"]],
            features=features,
            platform_types=data["platform_types"],
            confidence=data["confidence"],
            notes=data["notes"],
            last_updated=data["last_updated"],
            sample_signals=data["sample_signals"]
        )
    
    def match_score(self, signal: Signal) -> float:
        """
        Calculate how well a signal matches this signature.
        
        Args:
            signal: Signal to compare against this signature
            
        Returns:
            Match score between 0.0 (no match) and 1.0 (perfect match)
        """
        score = 0.0
        total_weight = 0.0
        
        # Frequency match (highest weight)
        weight = 3.0
        total_weight += weight
        if abs(signal.waveform.center_frequency - self.features.center_frequency) < self.features.bandwidth:
            freq_match = 1.0 - abs(signal.waveform.center_frequency - self.features.center_frequency) / self.features.bandwidth
            score += weight * max(0.0, freq_match)
        
        # Bandwidth match
        weight = 2.0
        total_weight += weight
        if signal.waveform.bandwidth > 0 and self.features.bandwidth > 0:
            bw_ratio = min(signal.waveform.bandwidth, self.features.bandwidth) / max(signal.waveform.bandwidth, self.features.bandwidth)
            score += weight * bw_ratio
        
        # Modulation match - Fix: use modulation_type instead of modulation
        weight = 2.0
        total_weight += weight
        if signal.waveform.modulation_type == self.features.modulation_type:
            score += weight
        
        # Power match
        weight = 1.0
        total_weight += weight
        if self.features.power_range and self.features.power_range[0] <= signal.power <= self.features.power_range[1]:
            score += weight
        elif self.features.typical_power and abs(signal.power - self.features.typical_power) < 10:
            power_match = 1.0 - abs(signal.power - self.features.typical_power) / 10.0
            score += weight * max(0.0, power_match)
        
        # Normalize score
        return score / total_weight if total_weight > 0 else 0.0


class RFSignatureDatabase:
    """Database for storing and retrieving RF signatures."""
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the RF signature database.
        
        Args:
            database_path: Path to the database file
        """
        if database_path is None:
            # Default to a database in the user's home directory
            self.database_path = Path.home() / ".helios" / "signatures" / "rf_signatures.json"
        else:
            self.database_path = Path(database_path)
            
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.signatures: Dict[str, RFSignature] = {}
        self.load_database()
        
    def load_database(self):
        """Load signatures from the database file."""
        if not self.database_path.exists():
            logger.info(f"No existing database found at {self.database_path}, creating new database")
            return
            
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                
            for sig_data in data.get("signatures", []):
                signature = RFSignature.from_dict(sig_data)
                self.signatures[signature.id] = signature
                
            logger.info(f"Loaded {len(self.signatures)} signatures from database")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def save_database(self):
        """Save signatures to the database file."""
        try:
            data = {
                "version": "1.0",
                "last_updated": time.time(),
                "signatures": [sig.to_dict() for sig in self.signatures.values()]
            }
            
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.signatures)} signatures to database")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_signature(self, signature: RFSignature):
        """Add a signature to the database."""
        self.signatures[signature.id] = signature
        self.save_database()
        
    def get_signature(self, signature_id: str) -> Optional[RFSignature]:
        """Get a signature by ID."""
        return self.signatures.get(signature_id)
    
    def remove_signature(self, signature_id: str) -> bool:
        """Remove a signature from the database."""
        if signature_id in self.signatures:
            del self.signatures[signature_id]
            self.save_database()
            return True
        return False
    
    def search_signatures(self, 
                         signature_type: Optional[SignatureType] = None,
                         platform_type: Optional[str] = None,
                         frequency_range: Optional[Tuple[float, float]] = None,
                         modulation_type: Optional[str] = None) -> List[RFSignature]:
        """
        Search for signatures matching the given criteria.
        
        Args:
            signature_type: Type of signature to search for
            platform_type: Platform type to search for
            frequency_range: Frequency range to search in
            modulation_type: Modulation type to search for
            
        Returns:
            List of matching signatures
        """
        results = []
        
        for signature in self.signatures.values():
            # Check signature type
            if signature_type and signature.signature_type != signature_type:
                continue
                
            # Check platform type
            if platform_type and platform_type not in signature.platform_types:
                continue
                
            # Check frequency range
            if frequency_range:
                sig_freq = signature.features.center_frequency
                if not (frequency_range[0] <= sig_freq <= frequency_range[1]):
                    continue
                    
            # Check modulation type
            if modulation_type and signature.features.modulation_type != modulation_type:
                continue
                
            results.append(signature)
            
        return results
    
    def identify_signal(self, signal: Signal, threshold: float = 0.7) -> List[Tuple[RFSignature, float]]:
        """
        Identify a signal by matching it against known signatures.
        
        Args:
            signal: Signal to identify
            threshold: Minimum match score to include in results
            
        Returns:
            List of (signature, match_score) tuples, sorted by score
        """
        matches = []
        
        for signature in self.signatures.values():
            score = signature.match_score(signal)
            if score >= threshold:
                matches.append((signature, score))
                
        # Sort by score, highest first
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def add_sample_signal(self, signature_id: str, signal: Signal):
        """
        Add a sample signal to a signature.
        
        Args:
            signature_id: ID of the signature
            signal: Sample signal to add
        """
        if signature_id not in self.signatures:
            logger.error(f"Signature {signature_id} not found")
            return
            
        # Convert signal to dictionary, keeping only essential information
        signal_data = {
            "id": signal.id,
            "center_frequency": signal.waveform.center_frequency,
            "bandwidth": signal.waveform.bandwidth,
            "modulation": signal.waveform.modulation_type,  # Fix: use modulation_type instead of modulation
            "power": signal.power,
            "timestamp": time.time()
        }
        
        self.signatures[signature_id].sample_signals.append(signal_data)
        self.signatures[signature_id].last_updated = time.time()
        self.save_database()