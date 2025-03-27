"""Data structures for network simulation and control."""

import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from helios.core.data_structures import Position, Platform

class NetworkNodeType(Enum):
    """Types of network nodes."""
    ROUTER = auto()
    SWITCH = auto()
    ENDPOINT = auto() # e.g., a device on a platform
    SERVER = auto()
    ACCESS_POINT = auto()

class LinkType(Enum):
    """Types of network links."""
    WIRED = auto()
    WIRELESS_RF = auto()
    FIBER_OPTIC = auto()
    SATELLITE = auto()

@dataclass
class NetworkNode:
    """Represents a node in the network."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NetworkNodeType = NetworkNodeType.ENDPOINT
    name: str = "Unnamed Node"
    platform_id: Optional[str] = None # Link to a platform if node is mobile/on platform
    position: Optional[Position] = None # Fixed position if not on a platform
    # Add specific properties like routing tables, buffers, processing power later
    # For control: configuration parameters
    config: Dict[str, Any] = field(default_factory=dict)
    
    # State variables
    operational_status: bool = True # Is the node functional?

    
@dataclass
class NetworkLink:
    """Represents a communication link between two network nodes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: LinkType = LinkType.WIRELESS_RF
    node1_id: str = field(default_factory=lambda: str(uuid.uuid4())) # ID of the first connected node
    node2_id: str = field(default_factory=lambda: str(uuid.uuid4())) # ID of the second connected node
    bandwidth: float = 1e6 # Bits per second
    latency: float = 0.01 # Seconds
    loss_rate: float = 0.0 # Packet loss probability (0-1)
    # For RF links, can add frequency, antenna gain etc.
    
    # Constraints / Policies for control
    max_bandwidth: Optional[float] = None
    qos_priority: int = 0 # Quality of Service priority
    allowed_protocols: List[str] = field(default_factory=list)
    
    # State variables
    current_utilization: float = 0.0 # Fraction of bandwidth used (0-1)
    operational_status: bool = True # Is the link up?
    
@dataclass
class NetworkPacket:
    """Represents a data packet traversing the network."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    destination_node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    size: int = 0 # Bytes
    payload: Any = None # Actual data or representation
    creation_time: float = 0.0 # Simulation time when created
    
    # Tracking info
    current_node_id: Optional[str] = None
    path: List[str] = field(default_factory=list) # Nodes visited
    hop_count: int = 0
    total_latency: float = 0.0 # Accumulated latency

@dataclass
class NetworkState:
    """Snapshot of the network's state at a given time."""
    timestamp: float
    nodes: Dict[str, NetworkNode] = field(default_factory=dict)
    links: Dict[str, NetworkLink] = field(default_factory=dict)
    # Can add traffic flow information, routing tables etc. later

@dataclass
class NetworkEvent:
    """Base class for events occurring within the network simulation."""
    event_type: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PacketReceivedEvent(NetworkEvent):
    """Event indicating a packet has been successfully received."""
    event_type: str = "packet_received"
    packet_id: str = ""
    destination_node_id: str = ""
    latency: float = 0.0

@dataclass
class LinkStateChangeEvent(NetworkEvent):
    """Event indicating a change in link state (e.g., quality, failure)."""
    event_type: str = "link_state_change"
    link_id: str = ""
    new_state: Dict[str, Any] = field(default_factory=dict)

# Add other specific event types as needed (e.g., NodeStateChangeEvent, PacketDroppedEvent)