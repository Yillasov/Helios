"""Interfaces for Live, Virtual, Constructive (LVC) integration."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from helios.core.data_structures import Platform, Signal, Position, Orientation


class ILVCInterface(ABC):
    """Base interface for LVC integration."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the LVC interface with configuration.
        
        Args:
            config: Configuration parameters for the interface
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> bool:
        """Connect to the external LVC environment.
        
        Args:
            connection_params: Connection parameters (address, port, etc.)
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the external LVC environment.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def export_platform(self, platform: Platform) -> bool:
        """Export a platform to the LVC environment.
        
        Args:
            platform: Platform to export
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def import_platforms(self) -> List[Platform]:
        """Import platforms from the LVC environment.
        
        Returns:
            List of imported platforms
        """
        pass
    
    @abstractmethod
    def export_signal(self, signal: Signal) -> bool:
        """Export a signal to the LVC environment.
        
        Args:
            signal: Signal to export
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def import_signals(self) -> List[Signal]:
        """Import signals from the LVC environment.
        
        Returns:
            List of imported signals
        """
        pass
    
    @abstractmethod
    def update(self) -> bool:
        """Update the LVC interface, processing incoming/outgoing data.
        
        Returns:
            Success status
        """
        pass


class IDISInterface(ILVCInterface):
    """Interface for DIS (Distributed Interactive Simulation) integration."""
    
    @abstractmethod
    def set_exercise_id(self, exercise_id: int) -> None:
        """Set the DIS exercise ID.
        
        Args:
            exercise_id: Exercise identifier
        """
        pass
    
    @abstractmethod
    def set_site_id(self, site_id: int) -> None:
        """Set the DIS site ID.
        
        Args:
            site_id: Site identifier
        """
        pass
    
    @abstractmethod
    def set_application_id(self, application_id: int) -> None:
        """Set the DIS application ID.
        
        Args:
            application_id: Application identifier
        """
        pass
    
    @abstractmethod
    def create_entity_id(self, platform_id: str) -> Tuple[int, int, int]:
        """Create a DIS entity ID for a platform.
        
        Args:
            platform_id: Helios platform ID
            
        Returns:
            Tuple of (site_id, application_id, entity_id)
        """
        pass


class IHLAInterface(ILVCInterface):
    """Interface for HLA (High Level Architecture) integration."""
    
    @abstractmethod
    def join_federation(self, federation_name: str, federate_name: str) -> bool:
        """Join an HLA federation.
        
        Args:
            federation_name: Name of the federation to join
            federate_name: Name for this federate
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def resign_federation(self) -> bool:
        """Resign from the current federation.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def publish_object_class(self, class_name: str, attributes: List[str]) -> bool:
        """Publish an object class.
        
        Args:
            class_name: Name of the object class
            attributes: List of attribute names to publish
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def subscribe_object_class(self, class_name: str, attributes: List[str]) -> bool:
        """Subscribe to an object class.
        
        Args:
            class_name: Name of the object class
            attributes: List of attribute names to subscribe to
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def register_object_instance(self, class_name: str, instance_name: str) -> str:
        """Register an object instance.
        
        Args:
            class_name: Name of the object class
            instance_name: Name for the instance
            
        Returns:
            Object instance handle
        """
        pass
    
    @abstractmethod
    def update_attribute_values(self, instance_handle: str, attributes: Dict[str, Any]) -> bool:
        """Update attribute values for an object instance.
        
        Args:
            instance_handle: Object instance handle
            attributes: Dictionary of attribute values
            
        Returns:
            Success status
        """
        pass