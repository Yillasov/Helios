"""RF component library with parameterized models."""

from typing import Dict, List, Optional, Any, Type, Callable
import uuid
import numpy as np

from helios.design.rf_components import (
    RFComponent, Amplifier, Filter, Mixer, Oscillator, 
    Antenna, Attenuator, Switch, Circulator
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class ComponentLibrary:
    """Library of parameterized RF component models."""
    
    def __init__(self):
        """Initialize the component library."""
        self.component_templates: Dict[str, Dict[str, Any]] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default component templates."""
        # Amplifier templates
        self.register_template(
            "lna_generic", 
            component_class=Amplifier,
            parameters={
                "gain": 20.0,
                "noise_figure": 1.5,
                "p1db": 15.0,
                "oip3": 30.0,
                "frequency_range": (0.1e9, 6e9)
            },
            description="Generic Low Noise Amplifier"
        )
        
        self.register_template(
            "pa_generic", 
            component_class=Amplifier,
            parameters={
                "gain": 30.0,
                "noise_figure": 5.0,
                "p1db": 30.0,
                "oip3": 45.0,
                "frequency_range": (0.1e9, 3e9)
            },
            description="Generic Power Amplifier"
        )
        
        # Filter templates
        self.register_template(
            "lpf_1ghz", 
            component_class=Filter,
            parameters={
                "filter_type": "lowpass",
                "cutoff_frequency": 1e9,
                "order": 5,
                "insertion_loss": 1.0
            },
            description="1 GHz Low-Pass Filter"
        )
        
        self.register_template(
            "bpf_wifi", 
            component_class=Filter,
            parameters={
                "filter_type": "bandpass",
                "center_frequency": 2.45e9,
                "bandwidth": 100e6,
                "order": 3,
                "insertion_loss": 2.0
            },
            description="Wi-Fi Band-Pass Filter"
        )
        
        # Mixer templates
        self.register_template(
            "mixer_generic", 
            component_class=Mixer,
            parameters={
                "conversion_loss": 7.0,
                "lo_power": 10.0,
                "isolation": {
                    "lo_rf": 25.0,
                    "lo_if": 30.0,
                    "rf_if": 20.0
                },
                "frequency_range": {
                    "rf": (0.1e9, 6e9),
                    "lo": (0.1e9, 6e9),
                    "if": (0, 2e9)
                }
            },
            description="Generic RF Mixer"
        )
        
        # Antenna templates
        self.register_template(
            "patch_antenna", 
            component_class=Antenna,
            parameters={
                "gain": 6.0,
                "polarization": "linear",
                "center_frequency": 2.4e9,
                "bandwidth": 200e6,
                "vswr": 1.5,
                "radiation_pattern": "directional"
            },
            description="Patch Antenna (2.4 GHz)"
        )
        
        # Oscillator templates
        self.register_template(
            "vco_generic", 
            component_class=Oscillator,
            parameters={
                "frequency_range": (1e9, 2e9),
                "tuning_voltage": (0, 5),
                "output_power": 10.0,
                "phase_noise": {
                    "1kHz": -80,
                    "10kHz": -100,
                    "100kHz": -120
                }
            },
            description="Generic Voltage Controlled Oscillator"
        )
    
    def register_template(self, 
                         template_id: str, 
                         component_class: Type[RFComponent], 
                         parameters: Dict[str, Any],
                         description: str = ""):
        """
        Register a component template.
        
        Args:
            template_id: Unique identifier for the template
            component_class: RFComponent class to instantiate
            parameters: Default parameters for the component
            description: Human-readable description
        """
        if template_id in self.component_templates:
            logger.warning(f"Overwriting existing template: {template_id}")
            
        self.component_templates[template_id] = {
            "component_class": component_class,
            "parameters": parameters,
            "description": description
        }
        logger.debug(f"Registered component template: {template_id}")
    
    def create_component(self, 
                        template_id: str, 
                        component_id: Optional[str] = None,
                        name: Optional[str] = None,
                        **parameter_overrides) -> Optional[RFComponent]:
        """
        Create a component from a template with optional parameter overrides.
        
        Args:
            template_id: ID of the template to use
            component_id: Optional ID for the new component
            name: Optional name for the new component
            **parameter_overrides: Parameters to override from the template
            
        Returns:
            Instantiated RFComponent or None if template not found
        """
        if template_id not in self.component_templates:
            logger.error(f"Template not found: {template_id}")
            return None
            
        template = self.component_templates[template_id]
        component_class = template["component_class"]
        
        # Create component with optional ID and name
        if name is None:
            name = f"{template_id.replace('_', ' ').title()}"
            
        component = component_class(component_id=component_id, name=name)
        
        # Apply template parameters
        for param_name, param_value in template["parameters"].items():
            if hasattr(component, param_name):
                setattr(component, param_name, param_value)
            else:
                component.parameters[param_name] = param_value
        
        # Apply parameter overrides
        for param_name, param_value in parameter_overrides.items():
            if hasattr(component, param_name):
                setattr(component, param_name, param_value)
            else:
                component.parameters[param_name] = param_value
        
        logger.debug(f"Created component from template {template_id}: {component.name} (ID: {component.id})")
        return component
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates.
        
        Returns:
            List of template information dictionaries
        """
        return [
            {
                "id": template_id,
                "class": template["component_class"].__name__,
                "description": template["description"]
            }
            for template_id, template in self.component_templates.items()
        ]