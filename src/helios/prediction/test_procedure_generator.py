"""Test procedure generators for RF system designs."""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import datetime

from helios.design.rf_system_design import RFSystemDesign
from helios.prediction.performance_predictor import PerformanceMetrics
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class TestProcedureGenerator:
    """Generates test procedures for RF system designs."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the test procedure generator.
        
        Args:
            design: RF system design to generate test procedures for
        """
        self.design = design
        self.output_dir = Path.home() / ".helios" / "test_procedures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_basic_test_procedure(self, performance_metrics: Optional[PerformanceMetrics] = None) -> Dict[str, Any]:
        """Generate a basic test procedure for the RF system.
        
        Args:
            performance_metrics: Optional performance metrics to include in test criteria
            
        Returns:
            Test procedure as a dictionary
        """
        # Extract key components for testing
        transmitters = [c for c_id, c in self.design.components.items() 
                      if hasattr(c, 'gain') and c.__class__.__name__ in ['Amplifier', 'Transmitter']]
        
        receivers = [c for c_id, c in self.design.components.items() 
                   if hasattr(c, 'noise_figure') and c.__class__.__name__ in ['Amplifier', 'Receiver']]
        
        oscillators = [c for c_id, c in self.design.components.items() 
                     if c.__class__.__name__ == 'Oscillator']
        
        filters = [c for c_id, c in self.design.components.items() 
                 if c.__class__.__name__ == 'Filter']
        
        # Create test procedure structure
        procedure = {
            "test_id": f"TP-{self.design.id or 'default'}-{datetime.datetime.now().strftime('%Y%m%d')}",
            "design_name": self.design.name,
            "date_generated": datetime.datetime.now().isoformat(),
            "test_sections": [],
            "equipment_required": self._generate_equipment_list()
        }
        
        # Add power-on test
        procedure["test_sections"].append({
            "name": "Power-On Test",
            "description": "Verify system powers on correctly and draws expected current",
            "steps": [
                {"step": 1, "action": "Connect power supply to system", "expected": "No shorts detected"},
                {"step": 2, "action": "Apply power and measure current draw", 
                 "expected": f"Current within specification (±10% of design value)"},
                {"step": 3, "action": "Verify status indicators", "expected": "All indicators show normal operation"}
            ]
        })
        
        # Add frequency and tuning tests if oscillators exist
        if oscillators:
            osc_tests = {
                "name": "Frequency Generation Test",
                "description": "Verify oscillator frequency accuracy and stability",
                "steps": []
            }
            
            for i, osc in enumerate(oscillators):
                freq = getattr(osc, 'frequency', 1e9)  # Default to 1 GHz if not specified
                osc_tests["steps"].append({
                    "step": i+1, 
                    "action": f"Measure oscillator frequency at {osc.id}", 
                    "expected": f"Frequency = {freq/1e6:.3f} MHz ±100 ppm"
                })
                
                osc_tests["steps"].append({
                    "step": i+2, 
                    "action": f"Measure phase noise at 10 kHz offset", 
                    "expected": "Phase noise better than -90 dBc/Hz"
                })
            
            procedure["test_sections"].append(osc_tests)
        
        # Add filter tests if filters exist
        if filters:
            filter_tests = {
                "name": "Filter Response Test",
                "description": "Verify filter frequency response",
                "steps": []
            }
            
            for i, filt in enumerate(filters):
                if hasattr(filt, 'filter_type') and hasattr(filt, 'cutoff_frequency'):
                    filter_type = getattr(filt, 'filter_type', 'lowpass')
                    cutoff = getattr(filt, 'cutoff_frequency', 1e9)
                    
                    filter_tests["steps"].append({
                        "step": i+1, 
                        "action": f"Measure {filter_type} response of {filt.id}", 
                        "expected": f"3dB cutoff at {cutoff/1e6:.1f} MHz ±5%"
                    })
                    
                    filter_tests["steps"].append({
                        "step": i+2, 
                        "action": f"Measure stopband attenuation", 
                        "expected": "At least 40 dB attenuation in stopband"
                    })
            
            procedure["test_sections"].append(filter_tests)
        
        # Add transmitter tests
        if transmitters:
            tx_tests = {
                "name": "Transmitter Test",
                "description": "Verify transmitter performance",
                "steps": []
            }
            
            # Use performance metrics if available
            expected_power = "Within design specification"
            if performance_metrics and hasattr(performance_metrics, 'power_consumption'):
                expected_power = f"{performance_metrics.power_consumption:.1f}W ±10%"
            
            tx_tests["steps"] = [
                {"step": 1, "action": "Connect spectrum analyzer to transmitter output", 
                 "expected": "Analyzer shows valid signal"},
                {"step": 2, "action": "Measure output power", 
                 "expected": expected_power},
                {"step": 3, "action": "Measure harmonic content", 
                 "expected": "Harmonics at least 40 dB below carrier"}
            ]
            
            procedure["test_sections"].append(tx_tests)
        
        # Add receiver tests
        if receivers:
            rx_tests = {
                "name": "Receiver Test",
                "description": "Verify receiver performance",
                "steps": []
            }
            
            # Use performance metrics if available
            expected_snr = "Within design specification"
            if performance_metrics and hasattr(performance_metrics, 'snr'):
                expected_snr = f"{performance_metrics.snr:.1f} dB minimum"
            
            rx_tests["steps"] = [
                {"step": 1, "action": "Connect signal generator to receiver input", 
                 "expected": "System detects input signal"},
                {"step": 2, "action": "Measure minimum detectable signal", 
                 "expected": "MDS better than -100 dBm"},
                {"step": 3, "action": "Measure SNR with -50 dBm input", 
                 "expected": expected_snr}
            ]
            
            procedure["test_sections"].append(rx_tests)
        
        # Add system integration test
        procedure["test_sections"].append({
            "name": "System Integration Test",
            "description": "Verify complete system operation",
            "steps": [
                {"step": 1, "action": "Configure system for loopback test", 
                 "expected": "System ready for testing"},
                {"step": 2, "action": "Transmit test pattern", 
                 "expected": "Pattern transmitted successfully"},
                {"step": 3, "action": "Verify received pattern", 
                 "expected": "Pattern received with BER < 1e-6"}
            ]
        })
        
        return procedure
    
    def _generate_equipment_list(self) -> List[Dict[str, str]]:
        """Generate list of required test equipment based on design."""
        equipment = [
            {"name": "Power Supply", "model": "Keysight E3631A or equivalent", 
             "description": "DC power supply, 0-6V/0-25V, 5A/1A"}
        ]
        
        # Add spectrum analyzer if we have RF components
        if any(c.__class__.__name__ in ['Amplifier', 'Mixer', 'Oscillator'] 
              for c_id, c in self.design.components.items()):
            equipment.append({
                "name": "Spectrum Analyzer", 
                "model": "Keysight N9000B or equivalent",
                "description": "9 kHz to 6 GHz frequency range"
            })
        
        # Add network analyzer if we have filters
        if any(c.__class__.__name__ == 'Filter' for c_id, c in self.design.components.items()):
            equipment.append({
                "name": "Network Analyzer", 
                "model": "Keysight E5063A or equivalent",
                "description": "100 kHz to 6 GHz frequency range"
            })
        
        # Add signal generator
        equipment.append({
            "name": "Signal Generator", 
            "model": "Keysight N5182B or equivalent",
            "description": "9 kHz to 6 GHz frequency range"
        })
        
        # Add oscilloscope
        equipment.append({
            "name": "Oscilloscope", 
            "model": "Keysight DSOX1204G or equivalent",
            "description": "70 MHz to 200 MHz bandwidth"
        })
        
        return equipment
    
    def generate_compliance_test_procedure(self) -> Dict[str, Any]:
        """Generate a compliance test procedure for regulatory requirements.
        
        Returns:
            Compliance test procedure as a dictionary
        """
        procedure = {
            "test_id": f"CP-{self.design.id or 'default'}-{datetime.datetime.now().strftime('%Y%m%d')}",
            "design_name": self.design.name,
            "date_generated": datetime.datetime.now().isoformat(),
            "test_sections": [],
            "equipment_required": self._generate_equipment_list() + [
                {"name": "EMI Receiver", "model": "Rohde & Schwarz ESR or equivalent", 
                 "description": "9 kHz to 6 GHz EMI test receiver"},
                {"name": "Anechoic Chamber", "model": "N/A", 
                 "description": "RF shielded room for emissions testing"}
            ]
        }
        
        # Add emissions test
        procedure["test_sections"].append({
            "name": "Conducted Emissions Test",
            "description": "Verify conducted emissions meet regulatory requirements",
            "steps": [
                {"step": 1, "action": "Connect LISN to power input", 
                 "expected": "LISN properly connected"},
                {"step": 2, "action": "Measure conducted emissions from 150 kHz to 30 MHz", 
                 "expected": "Emissions below Class B limits per FCC Part 15/CISPR 22"}
            ]
        })
        
        # Add radiated emissions test
        procedure["test_sections"].append({
            "name": "Radiated Emissions Test",
            "description": "Verify radiated emissions meet regulatory requirements",
            "steps": [
                {"step": 1, "action": "Place system in anechoic chamber", 
                 "expected": "System positioned correctly"},
                {"step": 2, "action": "Measure radiated emissions from 30 MHz to 1 GHz", 
                 "expected": "Emissions below Class B limits per FCC Part 15/CISPR 22"},
                {"step": 3, "action": "Measure radiated emissions from 1 GHz to 6 GHz", 
                 "expected": "Emissions below applicable limits"}
            ]
        })
        
        return procedure
    
    def export_procedure(self, procedure: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export test procedure to a file.
        
        Args:
            procedure: Test procedure dictionary
            filename: Optional filename, will be auto-generated if not provided
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            filename = f"{procedure['test_id']}.json"
        
        file_path = self.output_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(procedure, f, indent=2)
        
        logger.info(f"Exported test procedure to {file_path}")
        return str(file_path)
    
    def generate_all_procedures(self, 
                              performance_metrics: Optional[PerformanceMetrics] = None) -> Dict[str, str]:
        """Generate all test procedures and save to files.
        
        Args:
            performance_metrics: Optional performance metrics to include in test criteria
            
        Returns:
            Dictionary mapping procedure types to file paths
        """
        basic_procedure = self.generate_basic_test_procedure(performance_metrics)
        compliance_procedure = self.generate_compliance_test_procedure()
        
        basic_path = self.export_procedure(basic_procedure)
        compliance_path = self.export_procedure(compliance_procedure)
        
        return {
            "basic": basic_path,
            "compliance": compliance_path
        }