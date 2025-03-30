"""Armament system design module for RF-based weapon systems."""

from typing import Dict, List, Optional, Any, Set, Tuple
import uuid
import networkx as nx

from helios.design.rf_system_design import RFSystemDesign
from helios.design.rf_components import RFComponent
from helios.armament.targeting import TargetingSystem
from helios.armament.guidance import GuidanceSystem
from helios.utils.logger import get_logger
from helios.security.auth import AuthManager, ResourceType, ClassificationLevel # Added imports

logger = get_logger(__name__)

class ArmamentDesign(RFSystemDesign):
    """Specialized RF system design for armament applications."""
    
    def __init__(self, 
                 design_id: Optional[str] = None, 
                 name: str = "Armament System Design", 
                 auth_manager: Optional[AuthManager] = None): # Added auth_manager
        """Initialize the armament system design.
        
        Args:
            design_id: Optional unique identifier for the design
            name: Human-readable name for the design
            auth_manager: Optional authentication manager for access control
        """
        super().__init__(design_id=design_id, name=name)
        # Default to UNCLASSIFIED, requires explicit setting via secure method
        self._classification = ClassificationLevel.UNCLASSIFIED 
        self.targeting_system: Optional[TargetingSystem] = None
        self.guidance_system: Optional[GuidanceSystem] = None
        self.effective_range: float = 0.0  # meters
        self.power_output: float = 0.0  # watts
        self.military_specifications: Dict[str, Any] = {}
        self.auth_manager = auth_manager # Store auth_manager if provided
        
    @property
    def classification(self) -> ClassificationLevel:
         """Get the current classification level."""
         # In a real system, reading might also require a check
         return self._classification

    def set_classification(self, token: str, classification: ClassificationLevel):
        """Set the classification level of this design with authorization check.
        
        Args:
            token: Authentication token for the operation
            classification: New classification level
        """
        if not self.auth_manager:
            logger.error("AuthManager not configured for this design. Cannot set classification.")
            return

        # Check permission and clearance
        if not self.auth_manager.check_permission_with_classification(
            token, 
            ResourceType.SYSTEM, # Assuming design falls under SYSTEM resource type
            "configure",         # Permission needed to configure classification
            classification      # Clearance must meet or exceed this level
        ):
            # Logging handled by AuthManager
            return

        valid_classifications = list(ClassificationLevel)
        if classification not in valid_classifications:
            logger.warning(f"Invalid classification level: {classification}")
            return
            
        self._classification = classification
        logger.info(f"User set classification to {classification.name} for design '{self.name}'")
        
    def add_military_specification(self, token: str, spec_name: str, spec_value: Any):
        """Add a military specification requirement with authorization check."""
        if not self.auth_manager:
             logger.error("AuthManager not configured. Cannot add specification.")
             return

        # Check permission based on the design's current classification
        if not self.auth_manager.check_permission_with_classification(
            token, 
            ResourceType.SYSTEM, 
            "write", # Permission to modify the design
            self._classification # Clearance required based on the design's classification
        ):
            return

        self.military_specifications[spec_name] = spec_value
        logger.debug(f"Added military specification {spec_name} to design {self.name}")
        
    def calculate_effective_range(self, token: Optional[str] = None) -> float:
         """Calculate the effective range based on components. (Optional token for future checks)"""
         # Potentially add a check here if accessing component details requires auth
         # if token and self.auth_manager:
         #    if not self.auth_manager.check_permission_with_classification(...): return 0.0

         # Simple calculation based on transmitter power and receiver sensitivity
         transmitters = [c for c_id, c in self.components.items() 
                       if hasattr(c, 'gain') and c.__class__.__name__ == 'Amplifier']
         
         tx_power = sum(getattr(c, 'gain', 0) for c in transmitters)
         self.effective_range = 1000.0 * (tx_power / 20.0) ** 0.5
         return self.effective_range