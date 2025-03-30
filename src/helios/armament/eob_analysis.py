"""Electronic Order of Battle (EOB) analysis tools for RF systems."""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
from datetime import datetime

from helios.core.data_structures import Position, Signal
from helios.utils.logger import get_logger
from helios.targets.rf_signature_db import RFSignatureDatabase, RFSignature, SignatureType

# Define our own collection models since we're not importing sigint_collection
@dataclass
class CollectionResult:
    """Result of a collection operation."""
    target_id: str
    timestamp: float
    collection_success: bool
    confidence: float  # 0-1
    snr: float  # dB
    geolocation_accuracy: Optional[float] = None  # meters
    collected_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CollectionPriority(Enum):
    """Priority levels for collection."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SIGINTTarget:
    """Target for collection."""
    target_id: str
    designation: str
    position: Optional[Position] = None
    frequency_range: Tuple[float, float] = (0, 0)  # Hz
    signal_type: str = "UNKNOWN"
    priority: CollectionPriority = CollectionPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

class SIGINTCollectionModel:
    """Simplified model for collection capabilities."""
    
    def __init__(self):
        """Initialize the collection model."""
        self.targets: Dict[str, SIGINTTarget] = {}
        
    def add_target(self, target: SIGINTTarget):
        """Add a target to the model."""
        self.targets[target.target_id] = target

logger = get_logger(__name__)

class EmitterConfidence(Enum):
    """Confidence levels for emitter identification."""
    UNCONFIRMED = auto()
    POSSIBLE = auto()
    PROBABLE = auto()
    CONFIRMED = auto()


class EmitterStatus(Enum):
    """Operational status of tracked emitters."""
    ACTIVE = auto()
    INACTIVE = auto()
    INTERMITTENT = auto()
    SUSPECTED = auto()
    DESTROYED = auto()


@dataclass
class EmitterRecord:
    """Record of an electronic emitter in the battlefield."""
    emitter_id: str
    designation: str
    emitter_type: str
    position: Optional[Position] = None
    position_accuracy: float = 1000.0  # meters
    frequency_range: Tuple[float, float] = (0, 0)  # Hz
    power_estimate: Optional[float] = None  # dBm
    confidence: EmitterConfidence = EmitterConfidence.UNCONFIRMED
    status: EmitterStatus = EmitterStatus.SUSPECTED
    first_observed: float = field(default_factory=lambda: datetime.now().timestamp())
    last_observed: float = field(default_factory=lambda: datetime.now().timestamp())
    observation_count: int = 1
    threat_level: int = 0  # 0-10 scale
    associated_unit: Optional[str] = None
    collection_sources: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElectronicOrderOfBattle:
    """Electronic Order of Battle tracking and analysis system."""
    
    def __init__(self, sigint_model: Optional[SIGINTCollectionModel] = None):
        """Initialize the EOB system.
        
        Args:
            sigint_model: Optional SIGINT collection model to integrate with
        """
        self.emitters: Dict[str, EmitterRecord] = {}
        self.sigint_model = sigint_model
        self.collection_history: List[CollectionResult] = []
        
    def add_emitter(self, emitter: EmitterRecord):
        """Add an emitter to the EOB.
        
        Args:
            emitter: Emitter record to add
        """
        self.emitters[emitter.emitter_id] = emitter
        logger.info(f"Added emitter to EOB: {emitter.designation} ({emitter.emitter_type})")
        
    def update_from_collection(self, collection_result: CollectionResult, source_id: str):
        """Update EOB based on SIGINT collection result.
        
        Args:
            collection_result: SIGINT collection result
            source_id: ID of the collection source
        """
        self.collection_history.append(collection_result)
        
        if not collection_result.collection_success:
            logger.debug(f"Collection unsuccessful for target {collection_result.target_id}")
            return
            
        # Get target ID from collection result
        target_id = collection_result.target_id
        
        # Check if we already have this emitter
        emitter_id = target_id  # Use target ID as emitter ID for simplicity
        
        if emitter_id in self.emitters:
            # Update existing emitter
            emitter = self.emitters[emitter_id]
            emitter.last_observed = collection_result.timestamp
            emitter.observation_count += 1
            emitter.collection_sources.add(source_id)
            
            # Update position if available
            if collection_result.collected_data.get("position"):
                pos_data = collection_result.collected_data["position"]
                emitter.position = Position(pos_data["x"], pos_data["y"], pos_data["z"])
                emitter.position_accuracy = collection_result.geolocation_accuracy or emitter.position_accuracy
                
            # Update status
            emitter.status = EmitterStatus.ACTIVE
            
            # Improve confidence with repeated observations
            if emitter.confidence == EmitterConfidence.UNCONFIRMED and emitter.observation_count >= 2:
                emitter.confidence = EmitterConfidence.POSSIBLE
            elif emitter.confidence == EmitterConfidence.POSSIBLE and emitter.observation_count >= 5:
                emitter.confidence = EmitterConfidence.PROBABLE
            elif emitter.confidence == EmitterConfidence.PROBABLE and emitter.observation_count >= 10:
                emitter.confidence = EmitterConfidence.CONFIRMED
                
            logger.info(f"Updated emitter {emitter_id}: {emitter.observation_count} observations, confidence {emitter.confidence.name}")
            
        else:
            # Create new emitter from collection
            if self.sigint_model and target_id in self.sigint_model.targets:
                # Get target info from SIGINT model
                target = self.sigint_model.targets[target_id]
                
                # Create position if available
                position = None
                position_accuracy = 1000.0
                if collection_result.collected_data.get("position"):
                    pos_data = collection_result.collected_data["position"]
                    position = Position(pos_data["x"], pos_data["y"], pos_data["z"])
                    position_accuracy = collection_result.geolocation_accuracy or 1000.0
                
                # Create new emitter record
                emitter = EmitterRecord(
                    emitter_id=emitter_id,
                    designation=target.designation,
                    emitter_type=target.signal_type,
                    position=position,
                    position_accuracy=position_accuracy,
                    frequency_range=target.frequency_range,
                    power_estimate=collection_result.collected_data.get("power"),
                    confidence=EmitterConfidence.UNCONFIRMED,
                    status=EmitterStatus.ACTIVE,
                    first_observed=collection_result.timestamp,
                    last_observed=collection_result.timestamp,
                    observation_count=1,
                    threat_level=target.priority.value * 2,  # Scale priority to threat level
                    collection_sources={source_id}
                )
                
                self.add_emitter(emitter)
    
    def update_emitter_status(self, current_time: float, inactive_threshold: float = 300.0):
        """Update status of all emitters based on last observation time.
        
        Args:
            current_time: Current timestamp
            inactive_threshold: Time in seconds after which an emitter is considered inactive
        """
        for emitter_id, emitter in self.emitters.items():
            time_since_last_obs = current_time - emitter.last_observed
            
            if emitter.status != EmitterStatus.DESTROYED:
                if time_since_last_obs > inactive_threshold:
                    if emitter.status == EmitterStatus.ACTIVE:
                        emitter.status = EmitterStatus.INACTIVE
                        logger.info(f"Emitter {emitter_id} marked as inactive")
    
    def get_active_emitters(self) -> List[EmitterRecord]:
        """Get list of currently active emitters.
        
        Returns:
            List of active emitter records
        """
        return [e for e in self.emitters.values() if e.status == EmitterStatus.ACTIVE]
    
    def get_emitters_by_type(self, emitter_type: str) -> List[EmitterRecord]:
        """Get emitters of a specific type.
        
        Args:
            emitter_type: Type of emitter to filter by
            
        Returns:
            List of matching emitter records
        """
        return [e for e in self.emitters.values() if e.emitter_type == emitter_type]
    
    def get_emitters_in_area(self, center: Position, radius: float) -> List[EmitterRecord]:
        """Get emitters within a specified area.
        
        Args:
            center: Center position of the area
            radius: Radius in meters
            
        Returns:
            List of emitter records in the area
        """
        result = []
        for emitter in self.emitters.values():
            if emitter.position:
                distance = center.distance_to(emitter.position)
                if distance <= radius:
                    result.append(emitter)
        return result
    
    def get_threat_assessment(self) -> Dict[str, int]:
        """Get threat assessment for all emitters.
        
        Returns:
            Dictionary mapping emitter IDs to threat levels (0-10)
        """
        return {eid: emitter.threat_level for eid, emitter in self.emitters.items()}
    
    def generate_eob_report(self) -> Dict[str, Any]:
        """Generate a comprehensive EOB report.
        
        Returns:
            Dictionary containing EOB report data
        """
        active_count = len(self.get_active_emitters())
        total_count = len(self.emitters)
        
        # Count by type
        type_counts = {}
        for emitter in self.emitters.values():
            type_counts[emitter.emitter_type] = type_counts.get(emitter.emitter_type, 0) + 1
            
        # Count by confidence
        confidence_counts = {}
        for emitter in self.emitters.values():
            conf_name = emitter.confidence.name
            confidence_counts[conf_name] = confidence_counts.get(conf_name, 0) + 1
            
        # High threat emitters
        high_threat = [e for e in self.emitters.values() if e.threat_level >= 7]
        
        return {
            "timestamp": datetime.now().timestamp(),
            "total_emitters": total_count,
            "active_emitters": active_count,
            "emitter_types": type_counts,
            "confidence_levels": confidence_counts,
            "high_threat_count": len(high_threat),
            "high_threat_emitters": [
                {
                    "id": e.emitter_id,
                    "designation": e.designation,
                    "type": e.emitter_type,
                    "threat_level": e.threat_level,
                    "status": e.status.name
                }
                for e in high_threat
            ]
        }


class EOBAnalyzer:
    """Analysis tools for Electronic Order of Battle data."""
    
    def __init__(self, eob: ElectronicOrderOfBattle):
        """Initialize the EOB analyzer.
        
        Args:
            eob: Electronic Order of Battle instance
        """
        self.eob = eob
        
    def identify_command_nodes(self) -> List[str]:
        """Identify likely command and control nodes based on emission patterns.
        
        Returns:
            List of emitter IDs that are likely command nodes
        """
        command_nodes = []
        
        for emitter_id, emitter in self.eob.emitters.items():
            # Command nodes typically have specific characteristics
            is_command_node = False
            
            # Check for command node indicators
            if emitter.emitter_type in ["COMMAND", "C2", "HEADQUARTERS"]:
                is_command_node = True
            elif "command" in emitter.designation.lower():
                is_command_node = True
            elif emitter.metadata.get("traffic_volume", 0) > 8:  # High traffic volume
                is_command_node = True
                
            if is_command_node:
                command_nodes.append(emitter_id)
                
        return command_nodes
    
    def identify_networks(self) -> Dict[str, List[str]]:
        """Identify communication networks based on frequency and timing patterns.
        
        Returns:
            Dictionary mapping network IDs to lists of member emitter IDs
        """
        networks = {}
        assigned_emitters = set()
        
        # Group by frequency range first
        freq_groups = {}
        for emitter_id, emitter in self.eob.emitters.items():
            freq_key = f"{emitter.frequency_range[0]:.1f}-{emitter.frequency_range[1]:.1f}"
            if freq_key not in freq_groups:
                freq_groups[freq_key] = []
            freq_groups[freq_key].append(emitter_id)
            
        # Create networks from frequency groups
        for i, (freq_key, members) in enumerate(freq_groups.items()):
            if len(members) > 1:  # Only create networks with multiple members
                network_id = f"network_{i+1}"
                networks[network_id] = members
                assigned_emitters.update(members)
                
        # Add singleton networks for unassigned emitters
        for emitter_id in self.eob.emitters:
            if emitter_id not in assigned_emitters:
                network_id = f"singleton_{emitter_id}"
                networks[network_id] = [emitter_id]
                
        return networks
    
    def estimate_coverage(self) -> Dict[str, float]:
        """Estimate coverage area of each emitter.
        
        Returns:
            Dictionary mapping emitter IDs to coverage radius in meters
        """
        coverage = {}
        
        for emitter_id, emitter in self.eob.emitters.items():
            # Basic coverage estimate based on emitter type and power
            base_coverage = 5000.0  # Default 5km
            
            # Adjust based on emitter type
            if emitter.emitter_type == "RADAR":
                base_coverage = 50000.0  # 50km for radar
            elif emitter.emitter_type == "COMMUNICATION":
                base_coverage = 10000.0  # 10km for comms
            elif emitter.emitter_type == "NAVIGATION":
                base_coverage = 100000.0  # 100km for navigation
                
            # Adjust based on power if available
            if emitter.power_estimate is not None:
                # Simple power scaling: double power (in linear terms) = 1.4x range
                power_factor = 10 ** ((emitter.power_estimate - 30) / 20)  # Normalized to 30dBm
                base_coverage *= max(0.1, min(10, power_factor))
                
            coverage[emitter_id] = base_coverage
            
        return coverage
    
    def identify_gaps(self, area_center: Position, area_radius: float) -> List[Position]:
        """Identify gaps in electronic coverage within an area.
        
        Args:
            area_center: Center position of the area
            area_radius: Radius of the area in meters
            
        Returns:
            List of positions representing coverage gaps
        """
        # Get emitters in the area
        emitters = self.eob.get_emitters_in_area(area_center, area_radius)
        
        # Get coverage for each emitter
        coverage_map = self.estimate_coverage()
        
        # Simple gap identification using a grid approach
        grid_size = area_radius / 5
        gaps = []
        
        # Create a grid of points
        for x in np.arange(-area_radius, area_radius, grid_size):
            for y in np.arange(-area_radius, area_radius, grid_size):
                # Skip points outside the circle
                if x**2 + y**2 > area_radius**2:
                    continue
                    
                # Create a position for this grid point
                pos = Position(
                    area_center.x + x,
                    area_center.y + y,
                    0  # Assume ground level
                )
                
                # Check if this point is covered by any emitter
                covered = False
                for emitter in emitters:
                    if emitter.position:
                        distance = pos.distance_to(emitter.position)
                        coverage_radius = coverage_map.get(emitter.emitter_id, 5000.0)
                        if distance <= coverage_radius:
                            covered = True
                            break
                            
                if not covered:
                    gaps.append(pos)
                    
        return gaps