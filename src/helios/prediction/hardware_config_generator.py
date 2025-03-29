"""Hardware configuration generators for RF system designs."""

from typing import Dict, Any, List, Optional, Union
import json
import os
from pathlib import Path

from helios.design.rf_system_design import RFSystemDesign
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class HardwareConfigGenerator:
    """Generates hardware configurations from RF system designs."""
    
    def __init__(self, design: RFSystemDesign):
        """Initialize the hardware configuration generator.
        
        Args:
            design: RF system design to generate configurations for
        """
        self.design = design
        self.output_dir = Path.home() / ".helios" / "hardware_configs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_usrp_config(self) -> Dict[str, Any]:
        """Generate configuration for USRP devices.
        
        Returns:
            USRP configuration dictionary
        """
        # Extract key parameters from design
        tx_components = [c for c_id, c in self.design.components.items() 
                        if hasattr(c, 'gain') and c.__class__.__name__ == 'Amplifier']
        
        rx_components = [c for c_id, c in self.design.components.items() 
                        if hasattr(c, 'noise_figure') and c.__class__.__name__ in ['Amplifier', 'Filter']]
        
        # Find frequency components
        freq_components = [c for c_id, c in self.design.components.items() 
                          if hasattr(c, 'frequency') and c.__class__.__name__ == 'Oscillator']
        
        # Default values
        center_freq = 915e6  # 915 MHz
        sample_rate = 5e6    # 5 MHz
        tx_gain = 20.0       # 20 dB
        rx_gain = 30.0       # 30 dB
        
        # Extract from design if available
        if freq_components:
            center_freq = getattr(freq_components[0], 'frequency', center_freq)
        
        if tx_components:
            tx_gain = sum(getattr(c, 'gain', 0) for c in tx_components)
        
        if rx_components:
            rx_gain = sum(getattr(c, 'gain', 0) for c in rx_components if hasattr(c, 'gain'))
        
        # Generate USRP configuration
        config = {
            "device_args": "type=b200",
            "clock_source": "internal",
            "time_source": "internal",
            "tx": {
                "center_freq": center_freq,
                "gain": min(tx_gain, 89.0),  # USRP B200 max gain
                "antenna": "TX/RX",
                "sample_rate": sample_rate,
                "bandwidth": sample_rate * 0.8,
            },
            "rx": {
                "center_freq": center_freq,
                "gain": min(rx_gain, 73.0),  # USRP B200 max gain
                "antenna": "RX2",
                "sample_rate": sample_rate,
                "bandwidth": sample_rate * 0.8,
            }
        }
        
        return config
    
    def generate_hackrf_config(self) -> Dict[str, Any]:
        """Generate configuration for HackRF devices.
        
        Returns:
            HackRF configuration dictionary
        """
        # Start with USRP config and adapt for HackRF
        usrp_config = self.generate_usrp_config()
        
        # HackRF specific adjustments
        config = {
            "tx": {
                "center_freq": usrp_config["tx"]["center_freq"],
                "sample_rate": min(usrp_config["tx"]["sample_rate"], 20e6),  # HackRF max 20 MHz
                "amp_enable": usrp_config["tx"]["gain"] > 40,
                "vga_gain": min(int(usrp_config["tx"]["gain"] / 2), 47),  # HackRF max VGA gain
                "bandwidth": min(usrp_config["tx"]["bandwidth"], 15e6),  # HackRF max bandwidth
            },
            "rx": {
                "center_freq": usrp_config["rx"]["center_freq"],
                "sample_rate": min(usrp_config["rx"]["sample_rate"], 20e6),
                "lna_gain": min(int(usrp_config["rx"]["gain"] / 3), 40),  # HackRF max LNA gain
                "vga_gain": min(int(usrp_config["rx"]["gain"] / 3), 62),  # HackRF max VGA gain
                "amp_enable": usrp_config["rx"]["gain"] > 40,
                "bandwidth": min(usrp_config["rx"]["bandwidth"], 15e6),
            }
        }
        
        return config
    
    def generate_rtlsdr_config(self) -> Dict[str, Any]:
        """Generate configuration for RTL-SDR devices.
        
        Returns:
            RTL-SDR configuration dictionary
        """
        # Start with USRP config and adapt for RTL-SDR
        usrp_config = self.generate_usrp_config()
        
        # RTL-SDR specific adjustments (receive only)
        config = {
            "rx": {
                "center_freq": usrp_config["rx"]["center_freq"],
                "sample_rate": min(usrp_config["rx"]["sample_rate"], 2.4e6),  # RTL-SDR max sample rate
                "gain": min(usrp_config["rx"]["gain"], 49.6),  # RTL-SDR max gain
                "ppm": 0,
                "direct_sampling": 0,  # 0=off, 1=I, 2=Q
            }
        }
        
        return config
    
    def generate_fpga_config(self) -> Dict[str, Any]:
        """Generate configuration for FPGA acceleration.
        
        Returns:
            FPGA configuration dictionary
        """
        # Determine which processing blocks need acceleration
        needs_fft = any(c.__class__.__name__ in ['SpectrumAnalyzer', 'FFTProcessor'] 
                       for c_id, c in self.design.components.items())
        
        needs_filter = any(c.__class__.__name__ == 'Filter' 
                          for c_id, c in self.design.components.items())
        
        # Generate FPGA configuration
        config = {
            "device_type": "xilinx_ultrascale",
            "bitstreams": [],
            "interfaces": {
                "axi_dma": {
                    "tx_channel": 0,
                    "rx_channel": 1,
                    "buffer_size": 32768
                }
            }
        }
        
        # Add required bitstreams
        if needs_fft:
            config["bitstreams"].append("fft_accelerator.bit")
        
        if needs_filter:
            config["bitstreams"].append("fir_filter.bit")
        
        return config
    
    def generate_all_configs(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Generate all hardware configurations and save to files.
        
        Args:
            output_dir: Optional directory to save configurations
            
        Returns:
            Dictionary mapping device types to file paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate configurations
        configs = {
            "usrp": self.generate_usrp_config(),
            "hackrf": self.generate_hackrf_config(),
            "rtlsdr": self.generate_rtlsdr_config(),
            "fpga": self.generate_fpga_config()
        }
        
        # Save to files
        file_paths = {}
        design_id = self.design.id or "default"
        
        for device_type, config in configs.items():
            file_name = f"{design_id}_{device_type}_config.json"
            file_path = self.output_dir / file_name
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            file_paths[device_type] = str(file_path)
            logger.info(f"Generated {device_type} configuration: {file_path}")
        
        return file_paths