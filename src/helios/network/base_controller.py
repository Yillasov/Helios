"""Base implementation of the network controller interface."""

from typing import Dict, List, Any, Optional

from helios.network.interfaces import INetworkController
from helios.network.data_structures import NetworkNode, NetworkLink, NetworkState
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class BaseNetworkController(INetworkController):
    """
    Base implementation of the network controller interface.
    Provides default implementations that can be extended by subclasses.
    """
    
    def __init__(self):
        """Initialize the base network controller."""
        self.nodes = {}  # Dictionary of nodes by ID
        self.links = {}  # Dictionary of links by ID
        self.active_policies = {}  # Active routing policies
    
    def initialize(self, nodes: List[NetworkNode], links: List[NetworkLink]):
        """Initialize the network model with nodes and links."""
        self.nodes = {node.id: node for node in nodes}
        self.links = {link.id: link for link in links}
        logger.info(f"Network controller initialized with {len(nodes)} nodes and {len(links)} links")
    
    def modify_link_parameters(self, link_id: str, params: Dict[str, Any]):
        """Modify parameters of a specific link."""
        if link_id not in self.links:
            logger.warning(f"Cannot modify unknown link: {link_id}")
            return
            
        for param, value in params.items():
            if hasattr(self.links[link_id], param):
                setattr(self.links[link_id], param, value)
                logger.info(f"Modified link {link_id} parameter {param} to {value}")
            else:
                logger.warning(f"Unknown link parameter: {param}")
    
    def modify_node_parameters(self, node_id: str, params: Dict[str, Any]):
        """Modify parameters of a specific node."""
        if node_id not in self.nodes:
            logger.warning(f"Cannot modify unknown node: {node_id}")
            return
            
        for param, value in params.items():
            if param == "config" and hasattr(self.nodes[node_id], "config"):
                self.nodes[node_id].config.update(value)
                logger.info(f"Updated node {node_id} config")
            elif hasattr(self.nodes[node_id], param):
                setattr(self.nodes[node_id], param, value)
                logger.info(f"Modified node {node_id} parameter {param} to {value}")
            else:
                logger.warning(f"Unknown node parameter: {param}")
    
    def set_routing_policy(self, policy_name: str, policy_details: Dict[str, Any]):
        """Set or update a network-wide or node-specific routing policy."""
        self.active_policies[policy_name] = policy_details
        logger.info(f"Set routing policy: {policy_name}")
    
    def apply_network_effect(self, effect_type: str, target_id: str, parameters: Dict[str, Any]):
        """Apply a simulated effect (e.g., jamming, link failure)."""
        logger.info(f"Applying {effect_type} effect to {target_id}")
        
        if effect_type == "link_failure" and target_id in self.links:
            self.links[target_id].operational_status = False
            logger.info(f"Applied link failure to {target_id}")
        
        elif effect_type == "node_failure" and target_id in self.nodes:
            self.nodes[target_id].operational_status = False
            logger.info(f"Applied node failure to {target_id}")
        
        else:
            logger.warning(f"Unsupported effect type: {effect_type} or unknown target: {target_id}")
    
    def get_control_state(self) -> Dict[str, Any]:
        """Get the current configuration/state managed by the controller."""
        return {
            "active_policies": self.active_policies,
            "node_count": len(self.nodes),
            "link_count": len(self.links)
        }
    
    def get_network_state(self, timestamp: float) -> NetworkState:
        """Get the current state of the network."""
        # Create a basic network state
        state = NetworkState(
            timestamp=timestamp,
            nodes=list(self.nodes.values()),
            links=list(self.links.values())
        )
        return state