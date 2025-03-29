"""API for circuit simulator integration."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class CircuitSimulatorInterface(ABC):
    """Base interface for circuit simulator integration."""
    
    @abstractmethod
    def run_simulation(self, netlist: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run circuit simulation.
        
        Args:
            netlist: Circuit netlist
            params: Simulation parameters
            
        Returns:
            Dictionary containing simulation results
        """
        pass
        
    @abstractmethod
    def generate_netlist(self, design_data: Dict[str, Any]) -> str:
        """Generate netlist from design data.
        
        Args:
            design_data: Dictionary containing design parameters
            
        Returns:
            Circuit netlist as string
        """
        pass

class MockCircuitSimulator(CircuitSimulatorInterface):
    """Mock circuit simulator for testing."""
    
    def run_simulation(self, netlist: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            's_parameters': {},
            'noise_figure': 0.0,
            'gain': 0.0
        }
        
    def generate_netlist(self, design_data: Dict[str, Any]) -> str:
        return f"* Generated netlist for {design_data.get('name', 'unnamed')}"