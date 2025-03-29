"""Automated workflow tools for RF system design to prototype specifications."""

from typing import Dict, Any, List, Optional
import json
import os

from helios.design.rf_system_design import RFSystemDesign
from helios.prediction.performance_predictor import PerformancePredictor, PerformanceMetrics
from helios.design.design_validation import DesignValidator
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class DesignWorkflow:
    """Automates the workflow from RF system design to prototype specifications."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the design workflow.
        
        Args:
            design: RF system design to process
        """
        self.design = design
        self.validator = DesignValidator(design)
        self.predictor = PerformancePredictor(design)
        self.results: Dict[str, Any] = {}
        
    def run_workflow(self, 
                    environment_conditions: Dict[str, Any],
                    operational_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete design-to-prototype workflow.
        
        Args:
            environment_conditions: Environmental conditions for performance prediction
            operational_parameters: Operational parameters for performance prediction
            
        Returns:
            Dictionary containing workflow results
        """
        # Step 1: Validate the design
        validation_issues = self.validator.validate()
        
        # Step 2: Predict performance
        performance = self.predictor.predict_performance(
            environment_conditions, 
            operational_parameters
        )
        
        # Step 3: Generate prototype specifications
        prototype_specs = self._generate_prototype_specs(performance)
        
        # Compile results
        self.results = {
            "design_id": self.design.id,
            "design_name": self.design.name,
            "validation": {
                "passed": not any(issue.severity.name in ["ERROR", "CRITICAL"] for issue in validation_issues),
                "issues": [{"component": issue.component_id, 
                           "message": issue.message, 
                           "severity": issue.severity.name} 
                          for issue in validation_issues]
            },
            "performance": {
                "snr": performance.snr,
                "throughput": performance.throughput,
                "bit_error_rate": performance.bit_error_rate,
                "power_consumption": performance.power_consumption,
                "range": performance.range
            },
            "prototype_specifications": prototype_specs
        }
        
        return self.results
    
    def _generate_prototype_specs(self, performance: PerformanceMetrics) -> Dict[str, Any]:
        """Generate prototype specifications based on design and predicted performance.
        
        Args:
            performance: Predicted performance metrics
            
        Returns:
            Dictionary of prototype specifications
        """
        specs = {
            "components": [],
            "power_requirements": {
                "supply_voltage": 12.0,  # V
                "estimated_current": performance.power_consumption / 12.0,  # A
                "power_margin": 1.5  # safety factor
            },
            "mechanical": {
                "estimated_size": self._estimate_physical_size(),
                "cooling_requirements": self._estimate_cooling_requirements(performance.power_consumption)
            },
            "testing": {
                "recommended_tests": self._recommend_tests(performance)
            }
        }
        
        # Add component specifications
        for comp_id, component in self.design.components.items():
            specs["components"].append({
                "id": comp_id,
                "type": component.__class__.__name__,
                "specifications": self._extract_component_specs(component)
            })
        
        return specs
    
    def _estimate_physical_size(self) -> Dict[str, float]:
        """Estimate physical size based on component count and types."""
        # Simple estimation based on component count
        component_count = len(self.design.components)
        
        # Base size plus additional space per component
        width = 10.0 + (component_count * 2.0)  # cm
        height = 5.0 + (component_count * 1.0)  # cm
        depth = 2.0  # cm
        
        return {"width_cm": width, "height_cm": height, "depth_cm": depth}
    
    def _estimate_cooling_requirements(self, power_consumption: float) -> Dict[str, Any]:
        """Estimate cooling requirements based on power consumption."""
        if power_consumption < 5.0:
            return {"type": "passive", "description": "Passive cooling sufficient"}
        elif power_consumption < 20.0:
            return {"type": "active_air", "description": "Small fan recommended"}
        else:
            return {"type": "active_forced", "description": "Forced air cooling required"}
    
    def _recommend_tests(self, performance: PerformanceMetrics) -> List[str]:
        """Recommend tests based on performance metrics."""
        tests = ["Power consumption verification", "Frequency response"]
        
        # Add specific tests based on performance
        if performance.snr < 15.0:
            tests.append("Noise figure optimization")
        
        if performance.bit_error_rate > 1e-5:
            tests.append("BER testing under various conditions")
            
        if performance.range < 1000:
            tests.append("Range extension evaluation")
            
        return tests
    
    def _extract_component_specs(self, component: Any) -> Dict[str, Any]:
        """Extract relevant specifications from a component."""
        specs = {}
        
        # Extract common attributes if they exist
        for attr in ["frequency_range", "gain", "noise_figure", "power", "impedance"]:
            if hasattr(component, attr):
                specs[attr] = getattr(component, attr)
                
        return specs
    
    def export_specifications(self, output_path: str) -> None:
        """Export the prototype specifications to a file.
        
        Args:
            output_path: Path to save the specifications
        """
        if not self.results:
            logger.warning("No results to export. Run workflow first.")
            return
            
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Exported prototype specifications to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export specifications: {e}")