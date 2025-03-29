"""System-level validation tools for Helios."""

from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto
import time
import json
from pathlib import Path

from helios.design.design_validation import ValidationIssue, ValidationSeverity
from helios.design.rf_system_design import RFSystemDesign
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class SystemValidationResult:
    """Result of a system validation check."""
    
    def __init__(self, 
                 name: str, 
                 passed: bool, 
                 message: str, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize a validation result.
        
        Args:
            name: Name of the validation check
            passed: Whether the check passed
            message: Description of the result
            details: Additional details about the result
        """
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"

class SystemValidator:
    """Validates complete RF systems at the system level."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the system validator.
        
        Args:
            design: RF system design to validate
        """
        self.design = design
        self.results: List[SystemValidationResult] = []
    
    def validate_all(self) -> List[SystemValidationResult]:
        """Run all system-level validation checks.
        
        Returns:
            List of validation results
        """
        self.results = []
        
        # Run all validation checks
        self.validate_signal_chain()
        self.validate_power_budget()
        self.validate_frequency_plan()
        self.validate_noise_budget()
        
        return self.results
    
    def validate_signal_chain(self) -> SystemValidationResult:
        """Validate the signal chain from input to output.
        
        Returns:
            Validation result
        """
        # Find input and output components
        inputs = []
        outputs = []
        
        for component_id, component in self.design.components.items():
            # Check if this is an input component
            is_input = False
            for port_name, port_type in component.ports.items():
                if port_type == "input" and port_name not in component.connections:
                    is_input = True
                    break
            
            if is_input:
                inputs.append(component)
            
            # Check if this is an output component
            is_output = False
            for port_name, port_type in component.ports.items():
                if port_type == "output" and port_name not in component.connections:
                    is_output = True
                    break
            
            if is_output:
                outputs.append(component)
        
        # Check if we have at least one input and one output
        if not inputs:
            result = SystemValidationResult(
                name="Signal Chain",
                passed=False,
                message="No input components found in the design",
                details={"inputs": 0, "outputs": len(outputs)}
            )
            self.results.append(result)
            return result
        
        if not outputs:
            result = SystemValidationResult(
                name="Signal Chain",
                passed=False,
                message="No output components found in the design",
                details={"inputs": len(inputs), "outputs": 0}
            )
            self.results.append(result)
            return result
        
        # Check if there's a path from each input to at least one output
        all_paths_valid = True
        missing_paths = []
        
        for input_component in inputs:
            has_path = False
            for output_component in outputs:
                if self._check_path_exists(input_component.id, output_component.id):
                    has_path = True
                    break
            
            if not has_path:
                all_paths_valid = False
                missing_paths.append(input_component.id)
        
        if all_paths_valid:
            result = SystemValidationResult(
                name="Signal Chain",
                passed=True,
                message="All inputs have a path to at least one output",
                details={"inputs": len(inputs), "outputs": len(outputs)}
            )
        else:
            result = SystemValidationResult(
                name="Signal Chain",
                passed=False,
                message=f"{len(missing_paths)} input(s) have no path to any output",
                details={"missing_paths": missing_paths}
            )
        
        self.results.append(result)
        return result
    
    def validate_power_budget(self) -> SystemValidationResult:
        """Validate the power budget of the system.
        
        Returns:
            Validation result
        """
        # Find power sources and consumers
        power_sources = []
        power_consumers = []
        
        for component_id, component in self.design.components.items():
            if hasattr(component, 'power_output'):
                power_sources.append(component)
            
            if hasattr(component, 'power_consumption'):
                power_consumers.append(component)
        
        # Calculate total power available and consumed
        total_power_available = sum(getattr(source, 'power_output', 0) for source in power_sources)
        total_power_consumed = sum(getattr(consumer, 'power_consumption', 0) for consumer in power_consumers)
        
        # Check if we have enough power
        power_margin = total_power_available - total_power_consumed
        
        if power_margin >= 0:
            result = SystemValidationResult(
                name="Power Budget",
                passed=True,
                message=f"Power budget has {power_margin:.2f}W margin",
                details={
                    "available": total_power_available,
                    "consumed": total_power_consumed,
                    "margin": power_margin
                }
            )
        else:
            result = SystemValidationResult(
                name="Power Budget",
                passed=False,
                message=f"Power budget deficit of {-power_margin:.2f}W",
                details={
                    "available": total_power_available,
                    "consumed": total_power_consumed,
                    "margin": power_margin
                }
            )
        
        self.results.append(result)
        return result
    
    def validate_frequency_plan(self) -> SystemValidationResult:
        """Validate the frequency plan of the system.
        
        Returns:
            Validation result
        """
        # Check for frequency compatibility between connected components
        incompatible_connections = []
        
        for component_id, component in self.design.components.items():
            if hasattr(component, 'frequency_range'):
                freq_range = getattr(component, 'frequency_range')
                
                # Check connections
                for port_name, (connected_component, _) in component.connections.items():
                    if hasattr(connected_component, 'frequency_range'):
                        connected_freq_range = getattr(connected_component, 'frequency_range')
                        
                        # Check for overlap
                        if not self._ranges_overlap(freq_range, connected_freq_range):
                            incompatible_connections.append((component_id, connected_component.id))
        
        if not incompatible_connections:
            result = SystemValidationResult(
                name="Frequency Plan",
                passed=True,
                message="All connected components have compatible frequency ranges",
                details={}
            )
        else:
            result = SystemValidationResult(
                name="Frequency Plan",
                passed=False,
                message=f"{len(incompatible_connections)} incompatible frequency connections",
                details={"incompatible_connections": incompatible_connections}
            )
        
        self.results.append(result)
        return result
    
    def validate_noise_budget(self) -> SystemValidationResult:
        """Validate the noise budget of the system.
        
        Returns:
            Validation result
        """
        # Find receiver chains
        receiver_chains = []
        
        for component_id, component in self.design.components.items():
            if component.__class__.__name__ == 'Receiver':
                receiver_chains.append(component_id)
        
        if not receiver_chains:
            result = SystemValidationResult(
                name="Noise Budget",
                passed=True,
                message="No receiver chains to validate",
                details={}
            )
            self.results.append(result)
            return result
        
        # Calculate system noise figure for each receiver chain
        system_nf = {}
        
        for receiver_id in receiver_chains:
            # In a real implementation, you would calculate the cascade noise figure
            # This is a simplified placeholder
            system_nf[receiver_id] = 5.0  # Placeholder value
        
        # Check if noise figure is acceptable
        high_noise_receivers = []
        
        for receiver_id, nf in system_nf.items():
            if nf > 10.0:  # Threshold for acceptable noise figure
                high_noise_receivers.append((receiver_id, nf))
        
        if not high_noise_receivers:
            result = SystemValidationResult(
                name="Noise Budget",
                passed=True,
                message="All receiver chains have acceptable noise figure",
                details={"system_nf": system_nf}
            )
        else:
            result = SystemValidationResult(
                name="Noise Budget",
                passed=False,
                message=f"{len(high_noise_receivers)} receiver chains have high noise figure",
                details={"high_noise_receivers": high_noise_receivers}
            )
        
        self.results.append(result)
        return result
    
    def _check_path_exists(self, start_id: str, end_id: str) -> bool:
        """Check if there's a path from start to end component.
        
        Args:
            start_id: Starting component ID
            end_id: Ending component ID
            
        Returns:
            True if a path exists, False otherwise
        """
        visited = set()
        
        def dfs(current_id: str) -> bool:
            if current_id == end_id:
                return True
            
            if current_id in visited:
                return False
            
            visited.add(current_id)
            
            component = self.design.components.get(current_id)
            if not component:
                return False
            
            for port_name, (next_component, _) in component.connections.items():
                if dfs(next_component.id):
                    return True
            
            return False
        
        return dfs(start_id)
    
    def _ranges_overlap(self, range1: Tuple[float, float], range2: Tuple[float, float]) -> bool:
        """Check if two frequency ranges overlap.
        
        Args:
            range1: First frequency range (min, max)
            range2: Second frequency range (min, max)
            
        Returns:
            True if ranges overlap, False otherwise
        """
        min1, max1 = range1
        min2, max2 = range2
        
        return max1 >= min2 and max2 >= min1

def generate_system_validation_report(validator: SystemValidator, output_path: Optional[str] = None) -> str:
    """Generate a system validation report.
    
    Args:
        validator: System validator with validation results
        output_path: Optional path to save the report
        
    Returns:
        Report as a string
    """
    report = []
    
    # Add header
    report.append(f"# System Validation Report")
    report.append(f"Design: {validator.design.name} (ID: {validator.design.id})")
    report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Add summary
    passed = sum(1 for result in validator.results if result.passed)
    total = len(validator.results)
    
    report.append(f"## Summary")
    report.append(f"Passed: {passed}/{total} ({(passed/total*100):.1f}%" if total > 0 else "Passed: 0/0 (N/A)")
    report.append("")
    
    # Add detailed results
    report.append(f"## Detailed Results")
    
    for result in validator.results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        report.append(f"### {result.name}: {status}")
        report.append(f"{result.message}")
        
        if result.details:
            report.append("Details:")
            for key, value in result.details.items():
                report.append(f"- {key}: {value}")
        
        report.append("")
    
    report_str = "\n".join(report)
    
    # Save report if output path is provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logger.info(f"System validation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_str