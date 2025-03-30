"""Specialized component library for military-grade RF components."""

from typing import Dict, Any, Type
from helios.design.component_library import ComponentLibrary
from helios.design.rf_components import (
    RFComponent, Amplifier, Oscillator, Antenna, Filter
)
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class MilitaryComponentLibrary(ComponentLibrary):
    """Library focused on military-grade RF component models."""

    def __init__(self):
        """Initialize the military component library."""
        super().__init__()  # Initialize the base library
        self.component_templates: Dict[str, Dict[str, Any]] = {} # Override base templates if needed
        self._register_military_templates()
        logger.info("Initialized Military Component Library")

    def _register_military_templates(self):
        """Register templates specific to military applications."""

        # Hardened Low Noise Amplifier (LNA) Template
        self.register_template(
            "lna_hardened_s_band",
            component_class=Amplifier,
            parameters={
                "gain": 25.0,  # dB
                "noise_figure": 1.2,  # dB
                "p1db": 18.0,  # dBm
                "oip3": 35.0,  # dBm
                "frequency_range": (2.0e9, 4.0e9),  # Hz (S-band)
                "operating_temp_range": (-55, 125), # degrees C
                "vibration_spec": "MIL-STD-810G Method 514.6",
                "radiation_hardness_tid": 100, # krad(Si) Total Ionizing Dose
                "mtbf": 100000 # hours Mean Time Between Failures
            },
            description="Hardened S-Band Low Noise Amplifier"
        )

        # Secure Frequency Synthesizer Template
        self.register_template(
            "synthesizer_secure_vhf_uhf",
            component_class=Oscillator, # Simplified representation
            parameters={
                "frequency": 1.5e9, # Center frequency Hz
                "tuning_range": (100e6, 3.0e9), # Hz (VHF/UHF coverage)
                "phase_noise_10khz": -110.0, # dBc/Hz @ 10kHz offset
                "spurious_level": -70.0, # dBc
                "switching_speed": 50e-6, # seconds (50 microseconds)
                "operating_temp_range": (-40, 85), # degrees C
                "security_features": ["AES-256 encryption", "Anti-tamper mesh"],
                "power_consumption": 5.0 # Watts
            },
            description="Secure VHF/UHF Frequency Synthesizer with Anti-Tamper"
        )

        # Radiation Hardened Antenna Template
        self.register_template(
            "antenna_radhard_gps",
            component_class=Antenna,
            parameters={
                "antenna_type": "patch",
                "frequency_range": (1.55e9, 1.61e9), # GPS L1/L2 bands
                "gain": 5.0, # dBi
                "polarization": "RHCP", # Right-Hand Circular Polarization
                "vswr": 1.5,
                "operating_temp_range": (-60, 150), # degrees C
                "radiation_hardness_tid": 500, # krad(Si)
                "radiation_hardness_see": "Immune to LET > 80 MeV-cm2/mg", # Single Event Effects
                "shock_spec": "MIL-STD-810G Method 516.6"
            },
            description="Radiation Hardened GPS Patch Antenna"
        )

        # High-Power GaN Amplifier Module Template
        self.register_template(
            "pa_gan_x_band",
            component_class=Amplifier,
            parameters={
                "gain": 40.0, # dB
                "noise_figure": 6.0, # dB
                "p_sat": 47.0, # dBm (50 Watts) Saturated Power
                "pae": 45.0, # % Power Added Efficiency
                "frequency_range": (8.0e9, 12.0e9), # Hz (X-band)
                "operating_temp_range": (-40, 85), # degrees C
                "vibration_spec": "MIL-STD-810G Method 514.6",
                "cooling_requirement": "Requires heatsink/forced air"
            },
            description="High-Power X-Band GaN Power Amplifier Module"
        )

        logger.info(f"Registered {len(self.component_templates)} military-grade component templates.")

# Example usage (optional, could be in a separate script)
if __name__ == "__main__":
    mil_lib = MilitaryComponentLibrary()

    print("Available Military Templates:")
    # Correctly access the component_templates dictionary
    for name, template in mil_lib.component_templates.items():
        print(f"- {name}: {template['description']}")

    # Create an instance of a hardened LNA
    # Ensure create_component exists in ComponentLibrary or add it
    # Assuming create_component is implemented in the base class or needs to be added
    try:
        hardened_lna = mil_lib.create_component("lna_hardened_s_band", component_id="lna_001", name="FrontEndLNA")
    except AttributeError:
         logger.error("The base ComponentLibrary does not implement 'create_component'.")
         # Fallback or alternative component creation logic if needed
         template = mil_lib.component_templates.get("lna_hardened_s_band")
         if template:
             ComponentClass = template['component_class']
             hardened_lna = ComponentClass(component_id="lna_001", name="FrontEndLNA", **template['parameters'])
         else:
             hardened_lna = None
             logger.error("Template 'lna_hardened_s_band' not found.")

    if hardened_lna:
        print(f"\nCreated Component: {hardened_lna.name} (ID: {hardened_lna.id})")
        print("Parameters:")
        # Access parameters directly if it's an RFComponent instance
        for key, value in hardened_lna.parameters.items():
            print(f"  - {key}: {value}")