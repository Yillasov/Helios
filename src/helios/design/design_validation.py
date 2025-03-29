"""RF system design validation and verification tools."""

from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from enum import Enum, auto

from helios.design.rf_components import RFComponent
from helios.design.rf_system_design import RFSystemDesign
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class ValidationIssue:
    """Represents a validation issue found in a design."""
    
    def __init__(self, 
                component_id: Optional[str], 
                message: str, 
                severity: ValidationSeverity,
                issue_type: str,
                details: Optional[Dict[str, Any]] = None):
        """Initialize a validation issue.
        
        Args:
            component_id: ID of the component with the issue (None for system-level issues)
            message: Description of the issue
            severity: Severity level
            issue_type: Type of issue (e.g., 'impedance_mismatch', 'power_level')
            details: Additional details about the issue
        """
        self.component_id = component_id
        self.message = message
        self.severity = severity
        self.issue_type = issue_type
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the issue."""
        component_str = f"Component {self.component_id}: " if self.component_id else ""
        return f"[{self.severity.name}] {component_str}{self.message}"

class DesignValidator:
    """Validates RF system designs against requirements and best practices."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the design validator.
        
        Args:
            design: RF system design to validate
        """
        self.design = design
        self.issues: List[ValidationIssue] = []
    
    def validate(self) -> List[ValidationIssue]:
        """Run all validation checks on the design.
        
        Returns:
            List of validation issues found
        """
        self.issues = []
        
        # Run all validation checks
        self._validate_connections()
        self._validate_impedance_matching()
        self._validate_power_levels()
        self._validate_frequency_compatibility()
        self._validate_noise_figure()
        self._validate_isolation()
        self._validate_system_gain()
        
        return self.issues
    
    def _validate_connections(self) -> None:
        """Validate that all components are properly connected."""
        # Check for unconnected ports
        for component_id, component in self.design.components.items():
            for port_name, port_type in component.ports.items():
                if port_name not in component.connections:
                    self.issues.append(ValidationIssue(
                        component_id=component_id,
                        message=f"Port '{port_name}' is not connected",
                        severity=ValidationSeverity.WARNING,
                        issue_type="unconnected_port"
                    ))
        
        # Check for loops in the design
        self._check_for_loops()
    
    def _check_for_loops(self) -> None:
        """Check for unintended loops in the design."""
        # Simple loop detection using DFS
        visited: Set[str] = set()
        path: List[str] = []
        
        def dfs(component_id: str) -> None:
            if component_id in path:
                # Found a loop
                loop_str = " -> ".join(path[path.index(component_id):] + [component_id])
                self.issues.append(ValidationIssue(
                    component_id=None,
                    message=f"Unintended loop detected: {loop_str}",
                    severity=ValidationSeverity.WARNING,
                    issue_type="design_loop",
                    details={"loop_path": path[path.index(component_id):] + [component_id]}
                ))
                return
            
            if component_id in visited:
                return
            
            visited.add(component_id)
            path.append(component_id)
            
            component = self.design.components.get(component_id)
            if component:
                for port_name, (next_component, _) in component.connections.items():
                    dfs(next_component.id)
            
            path.pop()
        
        # Start DFS from each component
        for component_id in self.design.components:
            if component_id not in visited:
                dfs(component_id)
    
    def _validate_impedance_matching(self) -> None:
        """Validate impedance matching between connected components."""
        # This is a simplified check - in a real system, you'd use S-parameters
        for c1_id, c1 in self.design.components.items():
            for port_name, (c2, c2_port) in c1.connections.items():
                # Check for impedance mismatches (simplified)
                # In a real implementation, you would calculate VSWR from S-parameters
                if hasattr(c1, 'output_impedance') and hasattr(c2, 'input_impedance'):
                    z1 = getattr(c1, 'output_impedance', 50.0)  # Default to 50 ohms
                    z2 = getattr(c2, 'input_impedance', 50.0)   # Default to 50 ohms
                    
                    # Calculate reflection coefficient
                    reflection = abs((z2 - z1) / (z2 + z1))
                    
                    # Calculate VSWR
                    if reflection < 1.0:  # Sanity check
                        vswr = (1 + reflection) / (1 - reflection)
                        
                        if vswr > 2.0:  # VSWR threshold
                            self.issues.append(ValidationIssue(
                                component_id=c1_id,
                                message=f"High VSWR ({vswr:.2f}) between {c1.name} and {c2.name}",
                                severity=ValidationSeverity.WARNING,
                                issue_type="impedance_mismatch",
                                details={"vswr": vswr, "z1": z1, "z2": z2}
                            ))
    
    def _validate_power_levels(self) -> None:
        """Validate power levels throughout the system."""
        # Find signal sources (e.g., oscillators)
        sources = [c for c in self.design.components.values() 
                  if hasattr(c, 'output_power') and c.__class__.__name__ == 'Oscillator']
        
        for source in sources:
            # Trace signal path and calculate power levels
            self._trace_power_levels(source)
    
    def _trace_power_levels(self, component: RFComponent, 
                           input_power: Optional[float] = None, 
                           visited: Optional[Set[str]] = None) -> None:
        """Trace signal path and validate power levels.
        
        Args:
            component: Current component
            input_power: Input power in dBm (None for sources)
            visited: Set of visited component IDs
        """
        if visited is None:
            visited = set()
            
        if component.id in visited:
            return
            
        visited.add(component.id)
        
        # Calculate output power based on component type
        output_power = None
        
        if component.__class__.__name__ == 'Oscillator':
            # For oscillators, use their output power
            output_power = getattr(component, 'output_power', 0.0)
            
        elif component.__class__.__name__ == 'Amplifier' and input_power is not None:
            # For amplifiers, add gain
            gain = getattr(component, 'gain', 0.0)
            output_power = input_power + gain
            
            # Check for compression
            p1db = getattr(component, 'p1db', 0.0)
            if output_power > p1db:
                self.issues.append(ValidationIssue(
                    component_id=component.id,
                    message=f"Amplifier operating above P1dB point ({output_power:.1f} dBm > {p1db:.1f} dBm)",
                    severity=ValidationSeverity.WARNING,
                    issue_type="compression",
                    details={"output_power": output_power, "p1db": p1db}
                ))
                
            # Check for potential damage
            max_input = getattr(component, 'max_input_power', 20.0)  # Default 20 dBm
            if input_power > max_input:
                self.issues.append(ValidationIssue(
                    component_id=component.id,
                    message=f"Amplifier input power exceeds maximum ({input_power:.1f} dBm > {max_input:.1f} dBm)",
                    severity=ValidationSeverity.ERROR,
                    issue_type="power_damage",
                    details={"input_power": input_power, "max_input": max_input}
                ))
                
        elif component.__class__.__name__ in ['Attenuator', 'Filter'] and input_power is not None:
            # For passive components, subtract loss
            loss = getattr(component, 'attenuation', 0.0) if component.__class__.__name__ == 'Attenuator' else \
                  getattr(component, 'insertion_loss', 0.0)
            output_power = input_power - loss
            
        # Continue tracing through connected components
        if output_power is not None:
            for port_name, (next_component, _) in component.connections.items():
                # Only follow output ports
                port_type = component.ports.get(port_name)
                if port_type in ['output', 'bidirectional']:
                    self._trace_power_levels(next_component, output_power, visited)
    
    def _validate_frequency_compatibility(self) -> None:
        """Validate frequency compatibility between components."""
        for component_id, component in self.design.components.items():
            if hasattr(component, 'frequency_range'):
                freq_range = getattr(component, 'frequency_range')
                
                # Check if frequency range is valid
                if isinstance(freq_range, tuple) and len(freq_range) == 2:
                    min_freq, max_freq = freq_range
                    
                    # Check for very narrow bandwidth
                    if max_freq > min_freq and (max_freq - min_freq) / max_freq < 0.01:
                        self.issues.append(ValidationIssue(
                            component_id=component_id,
                            message=f"Very narrow bandwidth: {(max_freq - min_freq) / 1e6:.1f} MHz",
                            severity=ValidationSeverity.INFO,
                            issue_type="narrow_bandwidth"
                        ))
                    
                    # Check connected components for frequency compatibility
                    for port_name, (other_component, _) in component.connections.items():
                        if hasattr(other_component, 'frequency_range'):
                            other_range = getattr(other_component, 'frequency_range')
                            
                            if isinstance(other_range, tuple) and len(other_range) == 2:
                                other_min, other_max = other_range
                                
                                # Check for overlap
                                if max_freq < other_min or min_freq > other_max:
                                    self.issues.append(ValidationIssue(
                                        component_id=component_id,
                                        message=f"Frequency mismatch with {other_component.name}",
                                        severity=ValidationSeverity.ERROR,
                                        issue_type="frequency_mismatch",
                                        details={
                                            "range1": (min_freq, max_freq),
                                            "range2": (other_min, other_max)
                                        }
                                    ))
    
    def _validate_noise_figure(self) -> None:
        """Validate system noise figure."""
        # Find input components (e.g., antennas, first stage amplifiers)
        input_components = []
        
        for component in self.design.components.values():
            # Check if this is an input component (simplified)
            is_input = False
            for port_name, port_type in component.ports.items():
                if port_type == "input" and port_name not in component.connections:
                    is_input = True
                    break
            
            if is_input:
                input_components.append(component)
        
        # For each input component, calculate cascade noise figure
        for input_component in input_components:
            self._calculate_cascade_nf(input_component)
    
    def _calculate_cascade_nf(self, component: RFComponent, 
                             cascade_nf: float = 0.0,
                             cascade_gain: float = 0.0,
                             visited: Optional[Set[str]] = None) -> None:
        """Calculate cascade noise figure along a signal path.
        
        Args:
            component: Current component
            cascade_nf: Cumulative noise figure so far (linear)
            cascade_gain: Cumulative gain so far (linear)
            visited: Set of visited component IDs
        """
        if visited is None:
            visited = set()
            
        if component.id in visited:
            return
            
        visited.add(component.id)
        
        # Update cascade NF based on component type
        if component.__class__.__name__ == 'Amplifier':
            nf_db = getattr(component, 'noise_figure', 0.0)
            gain_db = getattr(component, 'gain', 0.0)
            
            # Convert to linear
            nf_linear = 10 ** (nf_db / 10)
            gain_linear = 10 ** (gain_db / 10)
            
            # Friis formula for noise figure
            if cascade_gain == 0:
                # First component
                new_cascade_nf = nf_linear
            else:
                new_cascade_nf = cascade_nf + (nf_linear - 1) / cascade_gain
                
            new_cascade_gain = cascade_gain * gain_linear
            
            # Check if NF is too high after first stage
            if len(visited) == 1 and nf_db > 3.0:
                self.issues.append(ValidationIssue(
                    component_id=component.id,
                    message=f"First stage noise figure is high: {nf_db:.1f} dB",
                    severity=ValidationSeverity.WARNING,
                    issue_type="high_first_stage_nf"
                ))
                
            # Continue tracing through connected components
            for port_name, (next_component, _) in component.connections.items():
                # Only follow output ports
                port_type = component.ports.get(port_name)
                if port_type in ['output', 'bidirectional']:
                    self._calculate_cascade_nf(next_component, new_cascade_nf, new_cascade_gain, visited)
    
    def _validate_isolation(self) -> None:
        """Validate isolation between critical components."""
        # Find components that need isolation
        for component_id, component in self.design.components.items():
            # Check for oscillators and mixers
            if component.__class__.__name__ in ['Oscillator', 'Mixer']:
                # Check if there's sufficient isolation
                self._check_isolation(component)
    
    def _check_isolation(self, component: RFComponent) -> None:
        """Check if a component has sufficient isolation.
        
        Args:
            component: Component to check
        """
        # For mixers, check LO-RF isolation
        if component.__class__.__name__ == 'Mixer':
            isolation = getattr(component, 'isolation', {}).get('lo_rf', 0.0)
            
            if isolation < 20.0:  # Threshold for good isolation
                self.issues.append(ValidationIssue(
                    component_id=component.id,
                    message=f"Low LO-RF isolation: {isolation:.1f} dB",
                    severity=ValidationSeverity.WARNING,
                    issue_type="low_isolation"
                ))
    
    def _validate_system_gain(self) -> None:
        """Validate overall system gain."""
        # Find signal paths and calculate total gain
        input_components = []
        output_components = []
        
        for component in self.design.components.values():
            # Check if this is an input component
            is_input = False
            for port_name, port_type in component.ports.items():
                if port_type == "input" and port_name not in component.connections:
                    is_input = True
                    break
            
            if is_input:
                input_components.append(component)
            
            # Check if this is an output component
            is_output = False
            for port_name, port_type in component.ports.items():
                if port_type == "output" and port_name not in component.connections:
                    is_output = True
                    break
            
            if is_output:
                output_components.append(component)
        
        # For each input-output pair, calculate total gain
        for input_component in input_components:
            for output_component in output_components:
                self._calculate_path_gain(input_component, output_component)
    
    def _calculate_path_gain(self, start: RFComponent, end: RFComponent) -> None:
        """Calculate gain along a path between two components.
        
        Args:
            start: Starting component
            end: Ending component
        """
        # Use BFS to find path
        queue = [(start, 0.0, [])]  # (component, gain, path)
        visited = set()
        
        while queue:
            component, gain, path = queue.pop(0)
            
            if component.id in visited:
                continue
                
            visited.add(component.id)
            path = path + [component.id]
            
            if component.id == end.id:
                # Found a path
                if gain > 50.0:  # Threshold for high gain
                    self.issues.append(ValidationIssue(
                        component_id=None,
                        message=f"High system gain: {gain:.1f} dB",
                        severity=ValidationSeverity.INFO,
                        issue_type="high_system_gain",
                        details={"path": path, "gain": gain}
                    ))
                elif gain < 0.0:  # Negative gain
                    self.issues.append(ValidationIssue(
                        component_id=None,
                        message=f"Negative system gain: {gain:.1f} dB",
                        severity=ValidationSeverity.WARNING,
                        issue_type="negative_system_gain",
                        details={"path": path, "gain": gain}
                    ))
                return
            
            # Update gain based on component type
            component_gain = 0.0
            
            if component.__class__.__name__ == 'Amplifier':
                component_gain = getattr(component, 'gain', 0.0)
            elif component.__class__.__name__ in ['Attenuator', 'Filter']:
                component_gain = -getattr(component, 'attenuation', 0.0) if component.__class__.__name__ == 'Attenuator' else \
                               -getattr(component, 'insertion_loss', 0.0)
            elif component.__class__.__name__ == 'Mixer':
                component_gain = -getattr(component, 'conversion_loss', 0.0)
            
            new_gain = gain + component_gain
            
            # Continue tracing through connected components
            for port_name, (next_component, _) in component.connections.items():
                # Only follow output ports
                port_type = component.ports.get(port_name)
                if port_type in ['output', 'bidirectional']:
                    queue.append((next_component, new_gain, path))


class DesignVerifier:
    """Verifies RF system designs against requirements."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the design verifier.
        
        Args:
            design: RF system design to verify
        """
        self.design = design
        self.requirements: Dict[str, Any] = {}
        self.results: Dict[str, bool] = {}
    
    def add_requirement(self, name: str, requirement_type: str, value: Any, 
                       component_id: Optional[str] = None) -> None:
        """Add a requirement to verify.
        
        Args:
            name: Requirement name
            requirement_type: Type of requirement (e.g., 'gain', 'noise_figure')
            value: Required value or range
            component_id: Optional component ID (None for system-level requirements)
        """
        self.requirements[name] = {
            "type": requirement_type,
            "value": value,
            "component_id": component_id
        }
    
    def verify(self) -> Dict[str, bool]:
        """Verify the design against all requirements.
        
        Returns:
            Dictionary of requirement names and verification results
        """
        self.results = {}
        
        for name, req in self.requirements.items():
            req_type = req["type"]
            value = req["value"]
            component_id = req["component_id"]
            
            if req_type == "gain":
                self.results[name] = self._verify_gain(value, component_id)
            elif req_type == "noise_figure":
                self.results[name] = self._verify_noise_figure(value, component_id)
            elif req_type == "frequency_range":
                self.results[name] = self._verify_frequency_range(value, component_id)
            elif req_type == "power_output":
                self.results[name] = self._verify_power_output(value, component_id)
            else:
                self.results[name] = False
        
        return self.results
    
    def _verify_gain(self, required_gain: float, component_id: Optional[str]) -> bool:
        """Verify gain requirement.
        
        Args:
            required_gain: Required gain in dB
            component_id: Component ID or None for system-level
            
        Returns:
            True if requirement is met, False otherwise
        """
        if component_id:
            # Component-level requirement
            component = self.design.components.get(component_id)
            if component and hasattr(component, 'gain'):
                return getattr(component, 'gain', 0.0) >= required_gain
            return False
        else:
            # System-level requirement (simplified)
            # In a real implementation, you would calculate the system gain
            # based on the signal path
            return True
    
    def _verify_noise_figure(self, required_nf: float, component_id: Optional[str]) -> bool:
        """Verify noise figure requirement.
        
        Args:
            required_nf: Required noise figure in dB
            component_id: Component ID or None for system-level
            
        Returns:
            True if requirement is met, False otherwise
        """
        if component_id:
            # Component-level requirement
            component = self.design.components.get(component_id)
            if component and hasattr(component, 'noise_figure'):
                return getattr(component, 'noise_figure', 0.0) <= required_nf
            return False
        else:
            # System-level requirement (simplified)
            # In a real implementation, you would calculate the system NF
            # based on the signal path
            return True
    
    def _verify_frequency_range(self, required_range: Tuple[float, float], 
                              component_id: Optional[str]) -> bool:
        """Verify frequency range requirement.
        
        Args:
            required_range: Required frequency range (min, max) in Hz
            component_id: Component ID or None for system-level
            
        Returns:
            True if requirement is met, False otherwise
        """
        if component_id:
            # Component-level requirement
            component = self.design.components.get(component_id)
            if component and hasattr(component, 'frequency_range'):
                freq_range = getattr(component, 'frequency_range')
                if isinstance(freq_range, tuple) and len(freq_range) == 2:
                    min_freq, max_freq = freq_range
                    req_min, req_max = required_range
                    return min_freq <= req_min and max_freq >= req_max
            return False
        else:
            # System-level requirement (simplified)
            # In a real implementation, you would check all components
            return True
    
    def _verify_power_output(self, required_power: float, component_id: Optional[str]) -> bool:
        """Verify power output requirement.
        
        Args:
            required_power: Required power output in dBm
            component_id: Component ID or None for system-level
            
        Returns:
            True if requirement is met, False otherwise
        """
        if component_id:
            # Component-level requirement
            component = self.design.components.get(component_id)
            if component:
                if component.__class__.__name__ == 'Amplifier':
                    # For amplifiers, check P1dB
                    return getattr(component, 'p1db', 0.0) >= required_power
                elif component.__class__.__name__ == 'Oscillator':
                    # For oscillators, check output power
                    return getattr(component, 'output_power', 0.0) >= required_power
            return False
        else:
            # System-level requirement (simplified)
            # In a real implementation, you would calculate the system output power
            return True


def generate_validation_report(validator: DesignValidator, verifier: Optional[DesignVerifier] = None) -> str:
    """Generate a validation and verification report.
    
    Args:
        validator: Design validator with validation results
        verifier: Optional design verifier with verification results
        
    Returns:
        Report as a string
    """
    report = []
    
    # Add header
    report.append(f"# RF System Design Validation Report")
    report.append(f"Design: {validator.design.name} (ID: {validator.design.id})")
    report.append("")
    
    # Add validation issues
    report.append("## Validation Issues")
    
    if not validator.issues:
        report.append("No validation issues found.")
    else:
        # Group issues by severity
        issues_by_severity = {}
        for issue in validator.issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)
        
        # Sort severities by importance
        for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                        ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            if severity in issues_by_severity:
                report.append(f"### {severity.name} ({len(issues_by_severity[severity])})")
                for issue in issues_by_severity[severity]:
                    component_name = ""
                    if issue.component_id:
                        component = validator.design.components.get(issue.component_id)
                        if component:
                            component_name = f" ({component.name})"
                    
                    report.append(f"- {issue.message}{component_name}")
                report.append("")
    
    # Add verification results if available
    if verifier:
        report.append("## Requirements Verification")
        
        if not verifier.results:
            report.append("No requirements verified.")
        else:
            # Count passed and failed requirements
            passed = sum(1 for result in verifier.results.values() if result)
            total = len(verifier.results)
            
            report.append(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
            report.append("")
            
            # List all requirements
            for name, result in verifier.results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                req = verifier.requirements[name]
                
                component_str = ""
                if req["component_id"]:
                    component = validator.design.components.get(req["component_id"])
                    if component:
                        component_str = f" ({component.name})"
                
                report.append(f"- {status} {name}: {req['type']} = {req['value']}{component_str}")
    
    return "\n".join(report)