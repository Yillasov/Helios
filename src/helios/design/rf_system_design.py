"""Defines the structure for representing RF System Designs."""

from typing import Dict, List, Optional, Tuple, Any
import uuid
import networkx as nx  # Using networkx for graph representation of connections

from helios.design.rf_components import RFComponent
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class RFSystemDesign:
    """
    Represents an RF system design consisting of interconnected RF components.
    Provides methods to build and manage the design topology.
    """

    def __init__(self, design_id: Optional[str] = None, name: str = "RF System Design"):
        """
        Initialize the RF System Design.

        Args:
            design_id: Optional unique identifier for the design.
            name: Human-readable name for the design.
        """
        self.id = design_id or str(uuid.uuid4())
        self.name = name
        self.components: Dict[str, RFComponent] = {}
        # Use NetworkX for managing connections more robustly
        self.connection_graph = nx.Graph()
        logger.info(f"Created RF System Design: {self.name} (ID: {self.id})")

    def add_component(self, component: RFComponent):
        """
        Add an RF component to the system design.

        Args:
            component: The RFComponent instance to add.

        Raises:
            ValueError: If a component with the same ID already exists.
        """
        if component.id in self.components:
            raise ValueError(f"Component with ID {component.id} already exists in the design.")
        
        self.components[component.id] = component
        # Add component as a node in the graph
        self.connection_graph.add_node(component.id, component_instance=component)
        logger.debug(f"Added component '{component.name}' (ID: {component.id}) to design '{self.name}'.")

    def get_component(self, component_id: str) -> Optional[RFComponent]:
        """
        Retrieve a component by its ID.

        Args:
            component_id: The ID of the component to retrieve.

        Returns:
            The RFComponent instance, or None if not found.
        """
        return self.components.get(component_id)

    def connect_components(self, comp1_id: str, port1_name: str, comp2_id: str, port2_name: str):
        """
        Connect two components within the design via their specified ports.

        Args:
            comp1_id: ID of the first component.
            port1_name: Name of the port on the first component.
            comp2_id: ID of the second component.
            port2_name: Name of the port on the second component.

        Raises:
            ValueError: If components or ports are not found, or if connection is invalid.
        """
        comp1 = self.get_component(comp1_id)
        comp2 = self.get_component(comp2_id)

        if not comp1:
            raise ValueError(f"Component with ID {comp1_id} not found in the design.")
        if not comp2:
            raise ValueError(f"Component with ID {comp2_id} not found in the design.")

        # Use the RFComponent's connect method to handle validation and internal state
        try:
            comp1.connect(port_name=port1_name, other_component=comp2, other_port=port2_name)
            # Add an edge in the graph to represent the connection
            # Use composite port identifiers for clarity (e.g., 'comp1_id:port1_name')
            node1_port = f"{comp1_id}:{port1_name}"
            node2_port = f"{comp2_id}:{port2_name}"
            # We can add nodes representing ports if needed, or just edge attributes
            self.connection_graph.add_edge(comp1_id, comp2_id, port1=port1_name, port2=port2_name)
            logger.info(f"Connected {comp1.name}:{port1_name} to {comp2.name}:{port2_name} in design '{self.name}'.")
        except ValueError as e:
            logger.error(f"Failed to connect components: {e}")
            raise

    def get_connections(self) -> List[Tuple[str, str, str, str]]:
        """
        Get a list of all connections in the design.

        Returns:
            List of tuples: (comp1_id, port1_name, comp2_id, port2_name)
        """
        connections = []
        for u, v, data in self.connection_graph.edges(data=True):
            # u and v are component IDs
            port1 = data.get('port1', 'unknown')
            port2 = data.get('port2', 'unknown')
            connections.append((u, port1, v, port2))
        return connections

    def __str__(self) -> str:
        """String representation of the design."""
        component_list = "\n  ".join([f"{comp.name} (ID: {comp.id})" for comp in self.components.values()])
        connection_list = "\n  ".join([f"{c1}:{p1} <-> {c2}:{p2}" for c1, p1, c2, p2 in self.get_connections()])
        return (
            f"RF System Design: {self.name} (ID: {self.id})\n"
            f"Components:\n  {component_list or 'None'}\n"
            f"Connections:\n  {connection_list or 'None'}"
        )