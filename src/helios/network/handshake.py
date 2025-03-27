"""Handshaking protocol for coordinated physical layer changes between network nodes."""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import time
import uuid

from helios.network.data_structures import NetworkNode, NetworkLink
from helios.core.data_structures import CognitiveWaveform
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class HandshakeState(Enum):
    """States in the handshaking protocol."""
    IDLE = auto()
    REQUEST_SENT = auto()
    REQUEST_RECEIVED = auto()
    ACKNOWLEDGED = auto()
    CONFIRMED = auto()
    REJECTED = auto()
    TIMEOUT = auto()
    COMPLETED = auto()

class WaveformHandshake:
    """Manages handshaking for coordinated waveform changes between nodes."""
    
    def __init__(self, timeout_seconds: float = 2.0):
        """Initialize the handshake manager."""
        self.active_handshakes: Dict[str, Dict[str, Any]] = {}
        self.completed_handshakes: List[Dict[str, Any]] = []
        self.timeout_seconds = timeout_seconds
    
    def initiate_handshake(self, 
                          source_node_id: str, 
                          target_node_id: str, 
                          link_id: str,
                          proposed_waveform: CognitiveWaveform,
                          current_time: float) -> str:
        """
        Initiate a handshake for waveform change.
        
        Args:
            source_node_id: ID of the node initiating the change
            target_node_id: ID of the node that needs to agree
            link_id: ID of the link to modify
            proposed_waveform: New waveform parameters
            current_time: Current simulation time
            
        Returns:
            Handshake ID
        """
        handshake_id = str(uuid.uuid4())
        
        handshake = {
            "id": handshake_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "link_id": link_id,
            "proposed_waveform": proposed_waveform,
            "state": HandshakeState.REQUEST_SENT,
            "start_time": current_time,
            "timeout_time": current_time + self.timeout_seconds,
            "completion_time": None,
            "response": None
        }
        
        self.active_handshakes[handshake_id] = handshake
        logger.info(f"Initiated handshake {handshake_id} from {source_node_id} to {target_node_id} for link {link_id}")
        
        return handshake_id
    
    def receive_request(self, handshake_id: str, current_time: float) -> Dict[str, Any]:
        """
        Mark a handshake request as received by the target node.
        
        Args:
            handshake_id: ID of the handshake
            current_time: Current simulation time
            
        Returns:
            Handshake data or empty dict if not found
        """
        if handshake_id not in self.active_handshakes:
            logger.warning(f"Received unknown handshake request: {handshake_id}")
            return {}
            
        handshake = self.active_handshakes[handshake_id]
        handshake["state"] = HandshakeState.REQUEST_RECEIVED
        
        logger.info(f"Node {handshake['target_node_id']} received handshake request {handshake_id}")
        return handshake
    
    def acknowledge_request(self,
                           handshake_id: str,
                           can_adapt: bool,
                           response_data: Optional[Dict[str, Any]] = None,
                           current_time: Optional[float] = None) -> bool: # Changed type hint to Optional[float]
        """
        Acknowledge a handshake request with acceptance or rejection.

        Args:
            handshake_id: ID of the handshake
            can_adapt: Whether the target node can adapt to the proposed waveform
            response_data: Additional response data
            current_time: Current simulation time (optional) # Updated docstring

        Returns:
            True if acknowledgment was successful, False otherwise
        """
        if handshake_id not in self.active_handshakes:
            logger.warning(f"Cannot acknowledge unknown handshake: {handshake_id}")
            return False

        handshake = self.active_handshakes[handshake_id]

        # Update state based on response
        if can_adapt:
            handshake["state"] = HandshakeState.ACKNOWLEDGED
            logger.info(f"Node {handshake['target_node_id']} acknowledged handshake {handshake_id}")
        else:
            handshake["state"] = HandshakeState.REJECTED
            logger.info(f"Node {handshake['target_node_id']} rejected handshake {handshake_id}")

        # Store response data
        handshake["response"] = response_data or {}

        # Note: current_time is passed but not used in this specific function implementation.
        # If it were needed, we'd add handling for when it's None.

        return True
    
    def confirm_handshake(self, handshake_id: str, current_time: float) -> bool:
        """
        Confirm the handshake, finalizing the waveform change agreement.
        
        Args:
            handshake_id: ID of the handshake
            current_time: Current simulation time
            
        Returns:
            True if confirmation was successful, False otherwise
        """
        if handshake_id not in self.active_handshakes:
            logger.warning(f"Cannot confirm unknown handshake: {handshake_id}")
            return False
            
        handshake = self.active_handshakes[handshake_id]
        
        # Only acknowledged handshakes can be confirmed
        if handshake["state"] != HandshakeState.ACKNOWLEDGED:
            logger.warning(f"Cannot confirm handshake {handshake_id} in state {handshake['state']}")
            return False
        
        handshake["state"] = HandshakeState.CONFIRMED
        handshake["completion_time"] = current_time
        
        logger.info(f"Handshake {handshake_id} confirmed between nodes {handshake['source_node_id']} and {handshake['target_node_id']}")
        
        # Move to completed handshakes
        self.completed_handshakes.append(handshake)
        del self.active_handshakes[handshake_id]
        
        return True
    
    def check_timeouts(self, current_time: float) -> List[str]:
        """
        Check for and handle timed-out handshakes.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of timed-out handshake IDs
        """
        timed_out = []
        
        for handshake_id, handshake in list(self.active_handshakes.items()):
            if current_time > handshake["timeout_time"]:
                handshake["state"] = HandshakeState.TIMEOUT
                logger.warning(f"Handshake {handshake_id} timed out")
                timed_out.append(handshake_id)
                
                # Move to completed handshakes
                self.completed_handshakes.append(handshake)
                del self.active_handshakes[handshake_id]
        
        return timed_out
    
    def get_handshake_status(self, handshake_id: str) -> Tuple[HandshakeState, Dict[str, Any]]:
        """
        Get the current status of a handshake.
        
        Args:
            handshake_id: ID of the handshake
            
        Returns:
            Tuple of (state, handshake_data)
        """
        # Check active handshakes
        if handshake_id in self.active_handshakes:
            handshake = self.active_handshakes[handshake_id]
            return handshake["state"], handshake
        
        # Check completed handshakes
        for handshake in self.completed_handshakes:
            if handshake["id"] == handshake_id:
                return handshake["state"], handshake
        
        return HandshakeState.IDLE, {}