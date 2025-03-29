"""System-level effects prediction for RF environments."""

from dataclasses import dataclass, field
from enum import Enum, auto
# Update the typing import to include TYPE_CHECKING
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
import numpy as np

from helios.core.data_structures import Signal, System, Platform
from helios.effects.component_vulnerability import ComponentVulnerabilityModel, VulnerabilityLevel
from helios.effects.rf_effects import RFEffect, RFEffectType, RFEffectSeverity
from helios.utils.logger import get_logger

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from helios.effects.cascading_effects import CascadingEffectsSimulator

logger = get_logger(__name__)

class SystemEffectType(Enum):
    """Types of system-level effects."""
    DEGRADED_PERFORMANCE = auto()  # System operates with reduced capabilities
    PARTIAL_FAILURE = auto()       # Some subsystems fail
    COMPLETE_FAILURE = auto()      # Entire system fails
    INTERMITTENT_OPERATION = auto() # System operates unreliably

@dataclass
class SystemEffect:
    """Represents an effect on an entire system."""
    system_id: str
    effect_type: SystemEffectType
    severity: float  # 0.0 to 1.0
    duration: float  # seconds
    affected_components: List[str] = field(default_factory=list)
    description: str = ""
    recovery_time: float = 0.0  # Time needed to recover after effect ends

class SystemDependencyGraph:
    """Represents dependencies between components in a system."""
    
    def __init__(self):
        """Initialize the dependency graph."""
        self.dependencies: Dict[str, Set[str]] = {}  # component -> components it depends on
        self.dependents: Dict[str, Set[str]] = {}    # component -> components that depend on it
        
    def add_dependency(self, dependent: str, dependency: str):
        """Add a dependency relationship (dependent depends on dependency)."""
        if dependent not in self.dependencies:
            self.dependencies[dependent] = set()
        self.dependencies[dependent].add(dependency)
        
        if dependency not in self.dependents:
            self.dependents[dependency] = set()
        self.dependents[dependency].add(dependent)
    
    def get_dependencies(self, component_id: str) -> Set[str]:
        """Get all components that this component depends on."""
        return self.dependencies.get(component_id, set())
    
    def get_dependents(self, component_id: str) -> Set[str]:
        """Get all components that depend on this component."""
        return self.dependents.get(component_id, set())
    
    def get_critical_components(self) -> Set[str]:
        """Get components that many others depend on (high centrality)."""
        return {comp for comp, deps in self.dependents.items() if len(deps) > 2}

class SystemEffectsPredictor:
    """Predicts system-level effects based on component-level effects."""
    
    def __init__(self):
        """Initialize the system effects predictor."""
        self.system_models: Dict[str, System] = {}
        self.component_models: Dict[str, ComponentVulnerabilityModel] = {}
        self.dependency_graphs: Dict[str, SystemDependencyGraph] = {}
        
    def add_system(self, system: System, dependency_graph: Optional[SystemDependencyGraph] = None):
        """Add a system model with its dependency graph."""
        self.system_models[system.id] = system
        if dependency_graph:
            self.dependency_graphs[system.id] = dependency_graph
        else:
            self.dependency_graphs[system.id] = SystemDependencyGraph()
    
    def add_component(self, component: ComponentVulnerabilityModel, system_id: str):
        """Add a component to a system."""
        self.component_models[component.component_id] = component
        # Ensure the system exists
        if system_id not in self.system_models:
            logger.warning(f"System {system_id} not found, creating a placeholder")
            self.system_models[system_id] = System(id=system_id, name=f"System {system_id}")
            self.dependency_graphs[system_id] = SystemDependencyGraph()
    
    def predict_system_effects(self, 
                              component_effects: List[RFEffect], 
                              system_id: str) -> List[SystemEffect]:
        """
        Predict system-level effects based on component-level effects.
        
        Args:
            component_effects: List of component-level effects
            system_id: ID of the system to analyze
            
        Returns:
            List of predicted system-level effects
        """
        if system_id not in self.system_models:
            logger.warning(f"System {system_id} not found")
            return []
        
        if not component_effects:
            return []
        
        # Group effects by component
        effects_by_component: Dict[str, List[RFEffect]] = {}
        for effect in component_effects:
            if effect.component_id not in effects_by_component:
                effects_by_component[effect.component_id] = []
            effects_by_component[effect.component_id].append(effect)
        
        # Get dependency graph
        dependency_graph = self.dependency_graphs.get(system_id, SystemDependencyGraph())
        
        # Identify critical components affected
        critical_components = dependency_graph.get_critical_components()
        affected_critical = critical_components.intersection(effects_by_component.keys())
        
        # Calculate system-wide impact
        total_components = len(self.component_models)
        if total_components == 0:
            return []
            
        affected_count = len(effects_by_component)
        affected_ratio = affected_count / total_components
        
        # Calculate maximum severity among component effects
        max_severity = 0.0
        max_duration = 0.0
        for effects in effects_by_component.values():
            for effect in effects:
                severity_value = effect.severity.value if hasattr(effect.severity, 'value') else effect.severity
                # Convert severity value to float, handling both numeric and enum cases
                if isinstance(severity_value, (int, float)):
                    max_severity = max(max_severity, float(severity_value))
                elif hasattr(severity_value, 'value'):
                    max_severity = max(max_severity, float(severity_value.value))
                max_duration = max(max_duration, effect.duration)
        
        # Normalize severity to 0-1 range
        normalized_severity = max_severity / 5.0  # Assuming 5 severity levels
        
        # Determine system effect type based on affected components and severity
        system_effects = []
        
        if affected_critical and normalized_severity > 0.6:
            # Critical components affected with high severity
            effect = SystemEffect(
                system_id=system_id,
                effect_type=SystemEffectType.COMPLETE_FAILURE,
                severity=normalized_severity,
                duration=max_duration,
                affected_components=list(effects_by_component.keys()),
                description=f"System failure due to effects on critical components: {', '.join(affected_critical)}",
                recovery_time=max_duration * 2  # Recovery takes longer than the effect duration
            )
            system_effects.append(effect)
            
        elif affected_ratio > 0.5 or (affected_critical and normalized_severity > 0.3):
            # Many components affected or critical components with moderate severity
            effect = SystemEffect(
                system_id=system_id,
                effect_type=SystemEffectType.PARTIAL_FAILURE,
                severity=normalized_severity * 0.8,  # Slightly reduced severity for partial failure
                duration=max_duration,
                affected_components=list(effects_by_component.keys()),
                description=f"Partial system failure affecting {affected_count} components",
                recovery_time=max_duration * 1.5
            )
            system_effects.append(effect)
            
        elif affected_ratio > 0.2 or normalized_severity > 0.4:
            # Some components affected or moderate severity
            effect = SystemEffect(
                system_id=system_id,
                effect_type=SystemEffectType.DEGRADED_PERFORMANCE,
                severity=normalized_severity * 0.6,
                duration=max_duration,
                affected_components=list(effects_by_component.keys()),
                description=f"Degraded system performance due to effects on {affected_count} components",
                recovery_time=max_duration
            )
            system_effects.append(effect)
            
        elif affected_count > 0:
            # Few components affected with low severity
            effect = SystemEffect(
                system_id=system_id,
                effect_type=SystemEffectType.INTERMITTENT_OPERATION,
                severity=normalized_severity * 0.4,
                duration=max_duration,
                affected_components=list(effects_by_component.keys()),
                description=f"Intermittent system operation due to minor effects on {affected_count} components",
                recovery_time=max_duration * 0.5
            )
            system_effects.append(effect)
        
        return system_effects
    
    def apply_system_effects(self, platform: Platform, effects: List[SystemEffect], current_time: float):
        """
        Apply system effects to a platform.
        
        Args:
            platform: The platform containing the affected systems
            effects: System effects to apply
            current_time: Current simulation time
        """
        for effect in effects:
            if effect.system_id in platform.equipped_systems:
                system = platform.equipped_systems[effect.system_id]
                
                # Store effect in system parameters
                if "system_effects" not in system.parameters:
                    system.parameters["system_effects"] = []
                
                system.parameters["system_effects"].append({
                    "type": effect.effect_type.name,
                    "severity": effect.severity,
                    "start_time": current_time,
                    "end_time": current_time + effect.duration,
                    "description": effect.description,
                    "affected_components": effect.affected_components
                })
                
                # Update system operational status
                if effect.effect_type == SystemEffectType.COMPLETE_FAILURE:
                    system.parameters["operational"] = False
                elif effect.effect_type == SystemEffectType.PARTIAL_FAILURE:
                    system.parameters["operational"] = True
                    system.parameters["degraded"] = True
                elif effect.effect_type == SystemEffectType.DEGRADED_PERFORMANCE:
                    system.parameters["operational"] = True
                    system.parameters["degraded"] = True
                    system.parameters["performance_factor"] = 1.0 - effect.severity

    # Add this method to the SystemEffectsPredictor class
    # Update the return type annotation with quotes to make it a forward reference
    def create_cascading_simulator(self) -> 'CascadingEffectsSimulator':
        """Create a cascading effects simulator that uses this predictor."""
        from helios.effects.cascading_effects import CascadingEffectsSimulator
        return CascadingEffectsSimulator(self)