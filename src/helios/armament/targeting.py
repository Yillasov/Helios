"""Targeting systems for RF-based armament applications."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum, auto
import uuid

from helios.targets.recognition import TargetRecognitionAlgorithm
from helios.utils.logger import get_logger
from helios.security.auth import AuthManager, ResourceType, ClassificationLevel # Added imports

logger = get_logger(__name__)

class TargetPriority(Enum):
    """Priority levels for targets."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class TargetDesignation:
    """Represents a designated target for an armament system."""
    
    def __init__(self, target_id: str, position: Tuple[float, float, float], 
                 priority: TargetPriority = TargetPriority.MEDIUM):
        """Initialize a target designation.
        
        Args:
            target_id: Unique identifier for the target
            position: 3D position (x, y, z) in meters
            priority: Target priority level
        """
        self.id = target_id
        self.position = position
        self.priority = priority
        self.signature: Dict[str, Any] = {}
        self.timestamp = 0.0
        
    def update_position(self, position: Tuple[float, float, float], timestamp: float):
        """Update the target position.
        
        Args:
            position: New 3D position (x, y, z) in meters
            timestamp: Time of the position update
        """
        self.position = position
        self.timestamp = timestamp

class TargetingSystem:
    """RF-based targeting system for armament applications with access control."""
    
    def __init__(self, 
                 recognition_algorithm: Optional[TargetRecognitionAlgorithm] = None,
                 auth_manager: Optional[AuthManager] = None): # Added auth_manager
        """Initialize the targeting system.
        
        Args:
            recognition_algorithm: Algorithm for target recognition
            auth_manager: Optional authentication manager for access control
        """
        self.id = str(uuid.uuid4())
        self.recognition_algorithm = recognition_algorithm
        self.targets: Dict[str, TargetDesignation] = {}
        self.max_range = 10000.0  # meters
        self.accuracy = 5.0  # meters CEP (Circular Error Probable)
        self.auth_manager = auth_manager # Store auth_manager
        # Assume targeting system operates at SECRET level by default
        self._classification = ClassificationLevel.SECRET 
        
    def designate_target(self, 
                        token: str, # Added token
                        position: Tuple[float, float, float], 
                        priority: TargetPriority = TargetPriority.MEDIUM) -> Optional[str]:
        """Designate a new target with authorization check.
        
        Args:
            token: Authentication token for the operation
            position: 3D position (x, y, z) in meters
            priority: Target priority level
            
        Returns:
            Target ID if successful, None otherwise
        """
        if not self.auth_manager:
            logger.error("AuthManager not configured for TargetingSystem. Cannot designate target.")
            return None

        # Check permission to execute targeting actions at the system's classification level
        if not self.auth_manager.check_permission_with_classification(
            token,
            ResourceType.SYSTEM, # Or potentially a more specific 'TARGETING' ResourceType
            "execute", # Permission to perform targeting actions
            self._classification # Required clearance level
        ):
            logger.warning(f"Authorization failed for target designation.")
            return None

        target_id = str(uuid.uuid4())
        target = TargetDesignation(target_id, position, priority)
        self.targets[target_id] = target
        logger.info(f"User designated new target {target_id} at position {position}")
        return target_id
        
    def get_highest_priority_target(self, token: Optional[str] = None) -> Optional[TargetDesignation]:
         """Get the highest priority target. (Optional token for future checks)"""
         # Potentially add checks here
         # if token and self.auth_manager: ...
         
         if not self.targets:
             return None
         return max(self.targets.values(), key=lambda t: t.priority.value)