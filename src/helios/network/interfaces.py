"""Interfaces for network components and control."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

# Import the new event structure and existing data structures
from helios.network.data_structures import NetworkNode, NetworkLink, NetworkState, NetworkPacket, NetworkEvent

class INetworkModel(ABC):
    """Interface for network simulation models."""

    @abstractmethod
    def initialize(self, nodes: List[NetworkNode], links: List[NetworkLink]):
        """Initialize the network model with nodes and links."""
        pass

    @abstractmethod
    def send_packet(self, packet: NetworkPacket, start_time: float):
        """Initiate sending a packet from its source node."""
        pass

    @abstractmethod
    def update(self, current_time: float, time_delta: float) -> List[NetworkEvent]: # Updated return type
        """Update the network state over a time step. Returns network events."""
        pass

    @abstractmethod
    def get_network_state(self, timestamp: float) -> NetworkState:
        """Get the current state of the network."""
        pass

class INetworkController(ABC):
    """Interface for controlling network behavior."""
    
    @abstractmethod
    def modify_link_parameters(self, link_id: str, params: Dict[str, Any]):
        """Modify parameters of a specific link (e.g., bandwidth, latency)."""
        pass
        
    @abstractmethod
    def modify_node_parameters(self, node_id: str, params: Dict[str, Any]):
        """Modify parameters of a specific node (e.g., routing table, status)."""
        pass
        
    @abstractmethod
    def set_routing_policy(self, policy_name: str, policy_details: Dict[str, Any]):
        """Set or update a network-wide or node-specific routing policy."""
        pass
        
    @abstractmethod
    def apply_network_effect(self, effect_type: str, target_id: str, parameters: Dict[str, Any]):
        """Apply a simulated effect (e.g., jamming, link failure)."""
        pass

    @abstractmethod
    def get_control_state(self) -> Dict[str, Any]:
        """Get the current configuration/state managed by the controller."""
        pass