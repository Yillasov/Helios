"""Simple network controller implementation."""

from typing import Dict, List, Any, Optional
from copy import deepcopy

from helios.network.base_controller import BaseNetworkController
from helios.network.data_structures import NetworkNode, NetworkLink, NetworkState
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class SimpleNetworkController(BaseNetworkController):
    """
    Simple network controller for basic scenarios.
    Provides minimal functionality without cognitive adaptations.
    """
    
    def __init__(self):
        """Initialize the simple network controller."""
        super().__init__()
        self.metrics = {}  # Simple metrics tracking
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update network metrics."""
        self.metrics.update(metrics)
        logger.debug(f"Updated network metrics: {metrics.keys()}")
    
    def get_control_state(self) -> Dict[str, Any]:
        """Get the current configuration/state managed by the controller."""
        base_state = super().get_control_state()
        base_state.update({
            "metrics": self.metrics
        })
        return base_state
    
    def get_network_state(self, timestamp: float) -> NetworkState:
        """Get the current state of the network with metrics."""
        state = super().get_network_state(timestamp)
        
        # Since NetworkState doesn't have a metadata attribute,
        # we'll include metrics in the details of a special NetworkEvent
        # and add it to the state's nodes dictionary
        
        # Create a special node to hold metrics if it doesn't exist
        metrics_node = NetworkNode(
            id="metrics_node",
            name="Network Metrics",
            config={"metrics": deepcopy(self.metrics)}
        )
        
        # Add the metrics node to the state
        nodes_dict = dict(state.nodes)
        nodes_dict["metrics_node"] = metrics_node
        
        # Create a new NetworkState with the updated nodes
        updated_state = NetworkState(
            timestamp=state.timestamp,
            nodes=nodes_dict,
            links=state.links
        )
        
        return updated_state