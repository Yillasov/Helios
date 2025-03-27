"""Network control module for RF-aware network adaptation."""

from typing import Dict, List, Optional, Any, Tuple

from helios.network.data_structures import NetworkNode, NetworkLink, NetworkState
import numpy as np

from helios.network.interfaces import INetworkController
from helios.network.metrics import estimate_rf_impact_on_network
from helios.cognitive.engine import CognitiveEngine
from helios.core.data_structures import CognitiveWaveform, AdaptationGoal
from helios.utils.logger import get_logger
from helios.network.handshake import WaveformHandshake, HandshakeState

logger = get_logger(__name__)

class NetworkControl(INetworkController):
    """
    Network control module that adapts network behavior based on RF conditions.
    Allows nodes to request waveform changes via the CognitiveEngine.
    """
    
    def __init__(self, cognitive_engine: Optional[CognitiveEngine] = None):
        """Initialize the network control module."""
        self.cognitive_engine = cognitive_engine or CognitiveEngine()
        self.links: Dict[str, NetworkLink] = {}
        self.nodes: Dict[str, NetworkNode] = {}
        self.link_quality_thresholds = {
            "poor": 0.3,      # Below this is considered poor quality
            "moderate": 0.6,  # Below this is considered moderate quality
            "good": 0.8       # Below this is considered good quality
            # Above 0.8 is excellent
        }
        self.adaptation_requests: List[Dict[str, Any]] = []
        self.active_policies: Dict[str, Dict[str, Any]] = {}
        self.handshake_manager = WaveformHandshake()  # Add handshake manager
        
        # Add routing table and resource allocation tracking
        self.routing_tables: Dict[str, Dict[str, str]] = {}  # node_id -> {dest_id: next_hop_id}
        self.resource_allocations: Dict[str, Dict[str, float]] = {}  # link_id -> {resource: value}
        self.rf_environment_state: Dict[str, Any] = {}  # Store RF environment data
        
    def initialize(self, nodes: Dict[str, NetworkNode], links: Dict[str, NetworkLink]):
        """Initialize with network nodes and links."""
        self.nodes = nodes
        self.links = links
        logger.info(f"Network control initialized with {len(nodes)} nodes and {len(links)} links")
    
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
            if param == "config":
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
        
        # Apply policy effects if needed
        if policy_details.get("apply_immediately", False):
            self._apply_routing_policy(policy_name)
    
    def apply_network_effect(self, effect_type: str, target_id: str, parameters: Dict[str, Any]):
        """Apply a simulated effect (e.g., jamming, link failure)."""
        if effect_type == "link_failure":
            if target_id in self.links:
                self.links[target_id].operational_status = False
                logger.info(f"Applied link failure to {target_id}")
        
        elif effect_type == "node_failure":
            if target_id in self.nodes:
                self.nodes[target_id].operational_status = False
                logger.info(f"Applied node failure to {target_id}")
        
        elif effect_type == "jamming":
            # Simulate jamming by reducing link quality
            affected_links = self._find_links_near_target(target_id, parameters.get("radius", 100.0))
            for link_id in affected_links:
                # Reduce bandwidth and increase loss rate
                jamming_factor = parameters.get("intensity", 0.5)
                current_bw = self.links[link_id].bandwidth
                self.links[link_id].bandwidth = current_bw * (1.0 - jamming_factor)
                self.links[link_id].loss_rate += jamming_factor * 0.5  # Max 50% additional loss
                logger.info(f"Applied jamming to link {link_id}, reducing bandwidth to {self.links[link_id].bandwidth}")
    
    def get_control_state(self) -> Dict[str, Any]:
        """Get the current configuration/state managed by the controller."""
        return {
            "active_policies": self.active_policies,
            "adaptation_requests": self.adaptation_requests,
            "link_quality_thresholds": self.link_quality_thresholds
        }
    
    def request_waveform_adaptation(self, 
                                   link_id: str, 
                                   snr_db: float, 
                                   interference_level_dbm: float = -120.0,
                                   waveform: Optional[CognitiveWaveform] = None) -> bool:
        """
        Request waveform adaptation based on link quality.
        
        Args:
            link_id: ID of the link experiencing quality issues
            snr_db: Current SNR in dB
            interference_level_dbm: Current interference level in dBm
            waveform: Optional waveform to adapt (if None, will be looked up)
            
        Returns:
            True if adaptation request was accepted, False otherwise
        """
        if link_id not in self.links:
            logger.warning(f"Cannot request adaptation for unknown link: {link_id}")
            return False
        
        link = self.links[link_id]
        
        # Skip if link is not wireless RF
        if link.type.name != "WIRELESS_RF":
            logger.info(f"Link {link_id} is not wireless RF, skipping adaptation request")
            return False
            
        # Evaluate link quality
        metrics = estimate_rf_impact_on_network(link, snr_db, interference_level_dbm)
        link_quality = metrics["link_quality"]
        
        # Determine if adaptation is needed based on link quality
        if link_quality >= self.link_quality_thresholds["good"]:
            logger.debug(f"Link {link_id} quality is good ({link_quality:.2f}), no adaptation needed")
            return False
        
        # Create adaptation request
        request = {
            "link_id": link_id,
            "metrics": metrics,
            "timestamp": 0.0,  # Will be set by cognitive engine
            "status": "pending"
        }
        
        # Add to request queue
        self.adaptation_requests.append(request)
        
        # If waveform is provided, request adaptation immediately
        if waveform is not None and self.cognitive_engine is not None:
            # Set adaptation goals based on link quality
            if link_quality < self.link_quality_thresholds["poor"]:
                # Poor link quality - prioritize SNR
                if AdaptationGoal.MAXIMIZE_SNR not in waveform.adaptation_goals:
                    waveform.adaptation_goals.append(AdaptationGoal.MAXIMIZE_SNR)
            else:
                # Moderate link quality - focus on interference avoidance
                if AdaptationGoal.MINIMIZE_INTERFERENCE not in waveform.adaptation_goals:
                    waveform.adaptation_goals.append(AdaptationGoal.MINIMIZE_INTERFERENCE)
            
            # Request adaptation from cognitive engine
            adaptations = self.cognitive_engine.decide_adaptation(waveform)
            
            if adaptations:
                request["status"] = "completed"
                request["adaptations"] = adaptations
                logger.info(f"Requested and received waveform adaptations for link {link_id}: {adaptations}")
                return True
            else:
                request["status"] = "failed"
                logger.warning(f"No adaptations available for link {link_id}")
                return False
        
        logger.info(f"Queued adaptation request for link {link_id}")
        return True
    
    def _find_links_near_target(self, target_id: str, radius: float) -> List[str]:
        """Find links within radius of a target node or position."""
        # Simple implementation - in a real system this would use spatial indexing
        affected_links = []
        
        # If target is a node, use its position
        target_pos = None
        if target_id in self.nodes:
            target_pos = self.nodes[target_id].position
        
        if target_pos:
            for link_id, link in self.links.items():
                # Check if either endpoint of the link is within radius
                node1 = self.nodes.get(link.node1_id)
                node2 = self.nodes.get(link.node2_id)
                
                if node1 and node1.position and target_pos.distance_to(node1.position) <= radius:
                    affected_links.append(link_id)
                elif node2 and node2.position and target_pos.distance_to(node2.position) <= radius:
                    affected_links.append(link_id)
        
        return affected_links
    
    def _apply_routing_policy(self, policy_name: str):
        """Apply a routing policy to the network."""
        if policy_name not in self.active_policies:
            return
            
        policy = self.active_policies[policy_name]
        
        # Example: apply QoS priorities to links
        if policy.get("type") == "qos":
            priorities = policy.get("priorities", {})
            for link_id, priority in priorities.items():
                if link_id in self.links:
                    self.links[link_id].qos_priority = priority
    
    def coordinate_waveform_change(self, 
                                  source_node_id: str, 
                                  target_node_id: str, 
                                  link_id: str, 
                                  new_waveform: CognitiveWaveform,
                                  current_time: float) -> str:
        """
        Coordinate a waveform change between two nodes using handshaking.
        
        Args:
            source_node_id: ID of the node initiating the change
            target_node_id: ID of the node that needs to agree
            link_id: ID of the link to modify
            new_waveform: New waveform parameters
            current_time: Current simulation time
            
        Returns:
            Handshake ID for tracking the coordination
        """
        # Validate nodes and link
        if source_node_id not in self.nodes:
            logger.warning(f"Source node {source_node_id} not found")
            return ""
            
        if target_node_id not in self.nodes:
            logger.warning(f"Target node {target_node_id} not found")
            return ""
            
        if link_id not in self.links:
            logger.warning(f"Link {link_id} not found")
            return ""
            
        # Initiate handshake
        handshake_id = self.handshake_manager.initiate_handshake(
            source_node_id, target_node_id, link_id, new_waveform, current_time
        )
        
        logger.info(f"Initiated waveform change coordination: {handshake_id}")
        return handshake_id
    
    def process_handshake_response(self, 
                                  handshake_id: str, 
                                  can_adapt: bool,
                                  current_time: float) -> bool:
        """
        Process a response to a handshake request.
        
        Args:
            handshake_id: ID of the handshake
            can_adapt: Whether the target node can adapt to the proposed waveform
            current_time: Current simulation time
            
        Returns:
            True if the handshake was completed successfully
        """
        # Acknowledge the request
        if not self.handshake_manager.acknowledge_request(handshake_id, can_adapt, None, current_time):
            return False
            
        # If target can adapt, confirm the handshake
        if can_adapt:
            if self.handshake_manager.confirm_handshake(handshake_id, current_time):
                # Get handshake details
                _, handshake = self.handshake_manager.get_handshake_status(handshake_id)
                
                # Apply the waveform change to the link
                if handshake:
                    link_id = handshake.get("link_id")
                    waveform = handshake.get("proposed_waveform")
                    
                    logger.info(f"Applying coordinated waveform change to link {link_id}")
                    # Here you would apply the actual waveform change to the link
                    # This depends on how your system represents the link's waveform
                    
                    return True
        
        return False
    
    def update_handshakes(self, current_time: float) -> List[str]:
        """
        Update handshake states and check for timeouts.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of completed handshake IDs
        """
        # Check for timeouts
        timed_out = self.handshake_manager.check_timeouts(current_time)
        
        # Get completed handshakes
        completed = []
        for handshake in self.handshake_manager.completed_handshakes:
            if handshake["state"] == HandshakeState.CONFIRMED:
                completed.append(handshake["id"])
                
        return completed

    def update_rf_environment(self, rf_data: Dict[str, Any]):
        """
        Update the RF environment state data.
        
        Args:
            rf_data: Dictionary containing RF environment data (interference, spectrum usage, etc.)
        """
        self.rf_environment_state.update(rf_data)
        logger.info(f"Updated RF environment state with {len(rf_data)} parameters")
        
        # Trigger cognitive adaptation if significant changes detected
        if self._should_adapt_network():
            self._adapt_network_to_rf_environment()
    
    def _should_adapt_network(self) -> bool:
        """Determine if network adaptation is needed based on RF changes."""
        # Simple implementation - adapt if interference levels changed significantly
        if 'interference_levels' in self.rf_environment_state:
            # Check if any interference level is above threshold
            for location, level in self.rf_environment_state['interference_levels'].items():
                if level > -80:  # dBm threshold for high interference
                    return True
        return False
    
    def _adapt_network_to_rf_environment(self):
        """Adapt network parameters based on current RF environment."""
        logger.info("Adapting network to RF environment")
        
        # 1. Update routing based on RF conditions
        self._update_cognitive_routing()
        
        # 2. Allocate resources based on RF conditions
        self._allocate_network_resources()
        
        # 3. Apply any policy changes needed
        for policy_name in self.active_policies:
            self._apply_routing_policy(policy_name)
    
    def _update_cognitive_routing(self):
        """Update routing tables based on RF environment and link quality."""
        # Calculate link costs based on RF conditions
        link_costs = self._calculate_rf_aware_link_costs()
        
        # Run simplified Dijkstra's algorithm for each node
        for node_id in self.nodes:
            if node_id not in self.routing_tables:
                self.routing_tables[node_id] = {}
            
            # Find shortest paths from this node to all others
            distances, next_hops = self._dijkstra_shortest_paths(node_id, link_costs)
            
            # Update routing table for this node
            for dest_id, next_hop in next_hops.items():
                if next_hop is not None:
                    self.routing_tables[node_id][dest_id] = next_hop
        
        logger.info("Updated cognitive routing tables based on RF environment")
    
    def _calculate_rf_aware_link_costs(self) -> Dict[str, Dict[str, float]]:
        """Calculate link costs considering RF interference and quality."""
        link_costs = {}
        
        for node_id in self.nodes:
            link_costs[node_id] = {}
            
            # Find all links connected to this node
            for link_id, link in self.links.items():
                if link.node1_id == node_id or link.node2_id == node_id:
                    # Determine the other node
                    other_node = link.node2_id if link.node1_id == node_id else link.node1_id
                    
                    # Skip non-operational links
                    if not link.operational_status:
                        continue
                    
                    # Base cost is inverse of bandwidth (higher bandwidth = lower cost)
                    base_cost = 1.0 / (link.bandwidth + 1.0)  # Add 1 to avoid division by zero
                    
                    # Factor in latency
                    latency_factor = link.latency * 10.0  # Scale latency impact
                    
                    # Factor in loss rate
                    loss_factor = 1.0 + (link.loss_rate * 5.0)  # Higher loss = higher cost
                    
                    # Factor in RF interference if available
                    rf_factor = 1.0
                    if 'interference_levels' in self.rf_environment_state:
                        # Check if we have interference data for this link
                        link_location = f"{link.node1_id}_{link.node2_id}"
                        if link_location in self.rf_environment_state['interference_levels']:
                            interference = self.rf_environment_state['interference_levels'][link_location]
                            # Convert dBm to linear scale and normalize
                            interference_linear = 10 ** (interference / 10) / 1e-9  # Reference: 1 nW
                            rf_factor = 1.0 + min(interference_linear / 1e3, 10.0)  # Cap at 10x cost increase
                    
                    # Calculate final cost
                    cost = base_cost * latency_factor * loss_factor * rf_factor
                    
                    # Store in cost dictionary
                    link_costs[node_id][other_node] = cost
        
        return link_costs
    
    def _dijkstra_shortest_paths(self, start_node: str, link_costs: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Run Dijkstra's algorithm to find shortest paths from start_node.
        
        Args:
            start_node: Starting node ID
            link_costs: Dictionary of link costs between nodes
            
        Returns:
            Tuple of (distances, next_hops) dictionaries
        """
        # Initialize distances and next hops
        distances = {node_id: float('infinity') for node_id in self.nodes}
        distances[start_node] = 0
        next_hops: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}
        
        # Priority queue for Dijkstra's algorithm
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda node: distances[node])
            
            # If current distance is infinity, remaining nodes are unreachable
            if distances[current] == float('infinity'):
                break
                
            # Remove current from unvisited
            unvisited.remove(current)
            
            # Check neighbors of current node
            if current in link_costs:
                for neighbor, cost in link_costs[current].items():
                    if neighbor in unvisited:
                        # Calculate new distance
                        new_distance = distances[current] + cost
                        
                        # Update if new distance is shorter
                        if new_distance < distances[neighbor]:
                            distances[neighbor] = new_distance
                            
                            # Update next hop - fix the type error here
                            if current == start_node:
                                # This is a direct neighbor of the start node
                                next_hops[neighbor] = neighbor  # This is correct
                            else:
                                # For nodes further away, use the next hop from current
                                next_hops[neighbor] = next_hops[current]
        
        return distances, next_hops
    
    def _allocate_network_resources(self):
        """Allocate network resources based on RF environment and traffic needs."""
        # Initialize resource allocations if needed
        for link_id in self.links:
            if link_id not in self.resource_allocations:
                self.resource_allocations[link_id] = {
                    "bandwidth_fraction": 1.0,  # Default: use all available bandwidth
                    "priority": 1,              # Default: normal priority
                    "power_level": 0.0          # Default: no power adjustment (in dB)
                }
        
        # Get traffic demand for each link (simplified)
        traffic_demand = self._estimate_traffic_demand()
        
        # Calculate interference levels for each link
        interference_levels = {}
        if 'interference_levels' in self.rf_environment_state:
            interference_levels = self.rf_environment_state['interference_levels']
        
        # Allocate resources based on demand and interference
        total_demand = sum(traffic_demand.values()) if traffic_demand else 1.0
        
        for link_id, link in self.links.items():
            # Skip non-RF links
            if link.type.name != "WIRELESS_RF":
                continue
                
            # Get demand for this link
            demand = traffic_demand.get(link_id, 0.1)  # Default to 10% if unknown
            demand_fraction = demand / total_demand if total_demand > 0 else 0.1
            
            # Get interference for this link
            link_location = f"{link.node1_id}_{link.node2_id}"
            interference = interference_levels.get(link_location, -120.0)  # Default to low interference
            
            # Calculate bandwidth allocation based on demand
            # Higher demand gets more bandwidth
            bandwidth_fraction = min(1.0, max(0.1, demand_fraction * 2.0))
            
            # Calculate priority based on interference
            # Higher interference gets higher priority (for QoS)
            if interference > -70:  # High interference
                priority = 3  # High priority
            elif interference > -90:  # Medium interference
                priority = 2  # Medium priority
            else:
                priority = 1  # Normal priority
            
            # Calculate power adjustment based on interference
            # Higher interference requires more power
            power_adjustment = min(3.0, max(0.0, (interference + 120) / 20.0))
            
            # Store resource allocation
            self.resource_allocations[link_id] = {
                "bandwidth_fraction": bandwidth_fraction,
                "priority": priority,
                "power_level": power_adjustment
            }
            
            # Apply resource allocation to link
            self.modify_link_parameters(link_id, {
                "qos_priority": priority
            })
            
            logger.debug(f"Allocated resources for link {link_id}: {self.resource_allocations[link_id]}")
        
        logger.info("Completed network resource allocation based on RF environment")
    
    def _estimate_traffic_demand(self) -> Dict[str, float]:
        """Estimate traffic demand for each link based on current utilization."""
        traffic_demand = {}
        
        for link_id, link in self.links.items():
            # Use current utilization as a proxy for demand
            # In a real system, this would use historical data and predictions
            traffic_demand[link_id] = link.current_utilization + 0.1  # Add 0.1 to ensure non-zero values
        
        return traffic_demand
    
    def get_routing_table(self, node_id: str) -> Dict[str, str]:
        """
        Get the current routing table for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary mapping destination node IDs to next hop node IDs
        """
        if node_id not in self.routing_tables:
            return {}
        return self.routing_tables[node_id]
    
    def get_resource_allocation(self, link_id: str) -> Dict[str, float]:
        """
        Get the current resource allocation for a link.
        
        Args:
            link_id: ID of the link
            
        Returns:
            Dictionary of resource allocations
        """
        if link_id not in self.resource_allocations:
            return {}
        return self.resource_allocations[link_id]
