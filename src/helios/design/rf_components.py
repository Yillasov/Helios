"""RF component definitions for system design."""

from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np

class RFComponent:
    """Base class for all RF components."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "RF Component"):
        """Initialize the RF component.
        
        Args:
            component_id: Unique ID for the component (auto-generated if None)
            name: Human-readable name
        """
        self.id = component_id or str(uuid.uuid4())
        self.name = name
        self.ports: Dict[str, str] = {}  # port_name -> port_type
        self.connections: Dict[str, Tuple['RFComponent', str]] = {}  # port_name -> (component, port)
        self.parameters: Dict[str, Any] = {}
    
    def add_port(self, name: str, port_type: str):
        """Add a port to the component.
        
        Args:
            name: Port name
            port_type: Port type (input, output, bidirectional)
        """
        self.ports[name] = port_type
    
    def connect(self, port_name: str, other_component: 'RFComponent', other_port: str):
        """Connect this component to another component.
        
        Args:
            port_name: Port name on this component
            other_component: The other component to connect to
            other_port: Port name on the other component
        """
        if port_name not in self.ports:
            raise ValueError(f"Port {port_name} not found on {self.name}")
            
        if other_port not in other_component.ports:
            raise ValueError(f"Port {other_port} not found on {other_component.name}")
            
        # Check port types compatibility
        port_type = self.ports[port_name]
        other_port_type = other_component.ports[other_port]
        
        if port_type == "input" and other_port_type == "input":
            raise ValueError("Cannot connect input to input")
            
        if port_type == "output" and other_port_type == "output":
            raise ValueError("Cannot connect output to output")
            
        # Create the connection
        self.connections[port_name] = (other_component, other_port)
        other_component.connections[other_port] = (self, port_name)
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the component.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            S-parameter matrix
        """
        # Default implementation: identity matrix
        num_ports = len(self.ports)
        return np.eye(num_ports, dtype=complex)


class Amplifier(RFComponent):
    """RF amplifier component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Amplifier"):
        """Initialize the amplifier."""
        super().__init__(component_id, name)
        self.add_port("input", "input")
        self.add_port("output", "output")
        
        # Default amplifier parameters
        self.gain = 0.0  # dB
        self.noise_figure = 0.0  # dB
        self.p1db = 0.0  # dBm
        self.oip3 = 0.0  # dBm
        self.frequency_range = (0.0, 0.0)  # Hz
    
    def get_gain(self, frequency: float) -> float:
        """Get the gain at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Gain in dB
        """
        # Simple frequency-dependent gain model
        min_freq, max_freq = self.frequency_range
        
        if frequency < min_freq or frequency > max_freq:
            # Outside frequency range, apply roll-off
            if frequency < min_freq:
                rolloff_factor = 1 - min(1.0, (min_freq - frequency) / (0.1 * min_freq))
            else:
                rolloff_factor = 1 - min(1.0, (frequency - max_freq) / (0.1 * max_freq))
            return self.gain * rolloff_factor
        
        return self.gain
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the amplifier."""
        gain_linear = 10 ** (self.get_gain(frequency) / 20)  # Convert dB to linear
        
        # Simple 2-port S-parameter matrix for an amplifier
        s_params = np.zeros((2, 2), dtype=complex)
        s_params[0, 0] = 0.1  # S11: Input reflection
        s_params[0, 1] = 0.0  # S12: Reverse transmission (isolation)
        s_params[1, 0] = gain_linear  # S21: Forward transmission (gain)
        s_params[1, 1] = 0.1  # S22: Output reflection
        
        return s_params


class Filter(RFComponent):
    """RF filter component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Filter"):
        """Initialize the filter."""
        super().__init__(component_id, name)
        self.add_port("input", "input")
        self.add_port("output", "output")
        
        # Default filter parameters
        self.filter_type = "lowpass"  # lowpass, highpass, bandpass, bandstop
        self.cutoff_frequency = 0.0  # Hz (for lowpass/highpass)
        self.center_frequency = 0.0  # Hz (for bandpass/bandstop)
        self.bandwidth = 0.0  # Hz (for bandpass/bandstop)
        self.order = 1  # Filter order
        self.insertion_loss = 0.0  # dB
    
    def get_insertion_loss(self, frequency: float) -> float:
        """Calculate insertion loss at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Insertion loss in dB (positive value)
        """
        # Simplified filter response models
        if self.filter_type == "lowpass":
            if frequency <= self.cutoff_frequency:
                return self.insertion_loss
            else:
                # Simple roll-off model: 20*n dB/decade
                decades = np.log10(frequency / self.cutoff_frequency)
                return self.insertion_loss + 20 * self.order * decades
                
        elif self.filter_type == "highpass":
            if frequency >= self.cutoff_frequency:
                return self.insertion_loss
            else:
                # Simple roll-off model: 20*n dB/decade
                decades = np.log10(self.cutoff_frequency / frequency)
                return self.insertion_loss + 20 * self.order * decades
                
        elif self.filter_type == "bandpass":
            # Simple bandpass model
            if abs(frequency - self.center_frequency) <= self.bandwidth / 2:
                return self.insertion_loss
            else:
                # Distance from band edge in decades
                if frequency < self.center_frequency - self.bandwidth / 2:
                    band_edge = self.center_frequency - self.bandwidth / 2
                    decades = np.log10(band_edge / frequency)
                else:
                    band_edge = self.center_frequency + self.bandwidth / 2
                    decades = np.log10(frequency / band_edge)
                return self.insertion_loss + 20 * self.order * decades
                
        elif self.filter_type == "bandstop":
            # Simple bandstop model
            if abs(frequency - self.center_frequency) <= self.bandwidth / 2:
                # Distance from center in normalized bandwidth
                normalized_dist = abs(frequency - self.center_frequency) / (self.bandwidth / 2)
                # Higher attenuation near center
                return self.insertion_loss + 20 * self.order * (1 - normalized_dist)
            else:
                return self.insertion_loss
        
        return self.insertion_loss  # Default fallback
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the filter."""
        # Calculate insertion loss in linear scale
        loss_db = self.get_insertion_loss(frequency)
        loss_linear = 10 ** (-loss_db / 20)
        
        # Simple 2-port S-parameter matrix for a filter
        s_params = np.zeros((2, 2), dtype=complex)
        s_params[0, 0] = 0.05  # S11: Input reflection
        s_params[0, 1] = loss_linear  # S12: Reverse transmission
        s_params[1, 0] = loss_linear  # S21: Forward transmission
        s_params[1, 1] = 0.05  # S22: Output reflection
        
        return s_params


class Mixer(RFComponent):
    """RF mixer component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Mixer"):
        """Initialize the mixer."""
        super().__init__(component_id, name)
        self.add_port("rf_input", "input")
        self.add_port("lo_input", "input")
        self.add_port("if_output", "output")
        
        # Default mixer parameters
        self.conversion_loss = 7.0  # dB
        self.lo_power = 10.0  # dBm
        self.isolation = {
            "lo_rf": 25.0,  # dB
            "lo_if": 30.0,  # dB
            "rf_if": 20.0   # dB
        }
        self.frequency_range = {
            "rf": (0.0, 0.0),  # Hz
            "lo": (0.0, 0.0),  # Hz
            "if": (0.0, 0.0)   # Hz
        }
    
    def get_conversion_loss(self, rf_freq: float, lo_freq: float) -> float:
        """Calculate conversion loss for specific RF and LO frequencies.
        
        Args:
            rf_freq: RF frequency in Hz
            lo_freq: LO frequency in Hz
            
        Returns:
            Conversion loss in dB
        """
        # Check if frequencies are within range
        rf_min, rf_max = self.frequency_range["rf"]
        lo_min, lo_max = self.frequency_range["lo"]
        
        # Apply penalties for out-of-range frequencies
        loss = self.conversion_loss
        
        if rf_freq < rf_min or rf_freq > rf_max:
            loss += 3.0  # Add 3dB penalty for out-of-range RF
            
        if lo_freq < lo_min or lo_freq > lo_max:
            loss += 5.0  # Add 5dB penalty for out-of-range LO
            
        return loss
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the mixer.
        
        Note: Mixers are nonlinear devices, so S-parameters are a simplification.
        """
        # Simple 3-port S-parameter matrix for a mixer
        # This is a significant simplification as mixers are nonlinear
        s_params = np.zeros((3, 3), dtype=complex)
        
        # Reflection coefficients
        s_params[0, 0] = 0.2  # RF port reflection
        s_params[1, 1] = 0.2  # LO port reflection
        s_params[2, 2] = 0.2  # IF port reflection
        
        # Isolation between ports (in linear scale)
        s_params[0, 1] = 10 ** (-self.isolation["lo_rf"] / 20)  # LO to RF isolation
        s_params[1, 0] = 10 ** (-self.isolation["lo_rf"] / 20)  # RF to LO isolation
        
        s_params[0, 2] = 10 ** (-self.isolation["rf_if"] / 20)  # IF to RF isolation
        s_params[2, 0] = 10 ** (-self.isolation["rf_if"] / 20)  # RF to IF isolation
        
        s_params[1, 2] = 10 ** (-self.isolation["lo_if"] / 20)  # IF to LO isolation
        s_params[2, 1] = 10 ** (-self.isolation["lo_if"] / 20)  # LO to IF isolation
        
        return s_params


class Oscillator(RFComponent):
    """RF oscillator component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Oscillator"):
        """Initialize the oscillator."""
        super().__init__(component_id, name)
        self.add_port("output", "output")
        self.add_port("tune", "input")  # For VCOs
        
        # Default oscillator parameters
        self.frequency = 1e9  # Hz
        self.frequency_range = (0.0, 0.0)  # Hz (for VCOs)
        self.tuning_voltage = (0.0, 0.0)  # V (for VCOs)
        self.output_power = 0.0  # dBm
        self.phase_noise = {
            "1kHz": -80,    # dBc/Hz
            "10kHz": -100,  # dBc/Hz
            "100kHz": -120  # dBc/Hz
        }
    
    def get_frequency(self, tuning_voltage: Optional[float] = None) -> float:
        """Get the oscillator frequency.
        
        Args:
            tuning_voltage: Tuning voltage for VCOs
            
        Returns:
            Frequency in Hz
        """
        if tuning_voltage is None:
            return self.frequency
            
        # For VCOs, calculate frequency based on tuning voltage
        min_freq, max_freq = self.frequency_range
        min_voltage, max_voltage = self.tuning_voltage
        
        if min_voltage == max_voltage:
            return min_freq
            
        # Linear tuning model
        voltage_ratio = (tuning_voltage - min_voltage) / (max_voltage - min_voltage)
        voltage_ratio = max(0.0, min(1.0, voltage_ratio))  # Clamp to [0, 1]
        
        return min_freq + voltage_ratio * (max_freq - min_freq)
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the oscillator."""
        # Oscillators are active components that generate signals
        # This is a simplified model
        s_params = np.zeros((1, 1), dtype=complex)
        s_params[0, 0] = 0.1  # Output reflection coefficient
        
        return s_params


class Antenna(RFComponent):
    """RF antenna component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Antenna"):
        """Initialize the antenna."""
        super().__init__(component_id, name)
        self.add_port("feed", "bidirectional")
        
        # Default antenna parameters
        self.gain = 0.0  # dBi
        self.polarization = "linear"  # linear, circular, etc.
        self.center_frequency = 0.0  # Hz
        self.bandwidth = 0.0  # Hz
        self.vswr = 1.0  # Voltage Standing Wave Ratio
        self.radiation_pattern = "omnidirectional"  # omnidirectional, directional, etc.
        self.efficiency = 0.9  # Antenna efficiency (0-1)
    
    def get_gain(self, frequency: float, direction: Optional[Tuple[float, float]] = None) -> float:
        """Get the antenna gain in a specific direction.
        
        Args:
            frequency: Frequency in Hz
            direction: (azimuth, elevation) in degrees, None for boresight
            
        Returns:
            Gain in dBi
        """
        # Simple frequency-dependent gain model
        if abs(frequency - self.center_frequency) > self.bandwidth / 2:
            # Outside bandwidth, apply roll-off
            distance = abs(frequency - self.center_frequency) - self.bandwidth / 2
            rolloff_factor = 1 / (1 + (distance / (self.bandwidth / 2)) ** 2)
            freq_gain = self.gain * rolloff_factor
        else:
            freq_gain = self.gain
            
        # Apply directional pattern if direction is specified
        if direction is not None and self.radiation_pattern == "directional":
            azimuth, elevation = direction
            
            # Simple directional model: cosine pattern
            # Maximum gain at boresight (0, 0)
            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)
            
            # Calculate angular distance from boresight
            angular_factor = np.cos(azimuth_rad) * np.cos(elevation_rad)
            angular_factor = max(0.0, angular_factor)  # No negative gain
            
            return freq_gain * angular_factor
            
        return freq_gain
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the antenna."""
        # For an antenna, S-parameters represent the reflection coefficient
        # related to impedance matching
        
        # Calculate reflection coefficient from VSWR
        reflection = (self.vswr - 1) / (self.vswr + 1)
        
        # Apply frequency dependence
        if abs(frequency - self.center_frequency) > self.bandwidth / 2:
            # Worse matching outside bandwidth
            reflection = reflection + 0.2 * min(1.0, abs(frequency - self.center_frequency) / self.center_frequency)
            reflection = min(0.9, reflection)  # Cap at 0.9
            
        s_params = np.zeros((1, 1), dtype=complex)
        s_params[0, 0] = reflection
        
        return s_params


class Attenuator(RFComponent):
    """RF attenuator component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Attenuator"):
        """Initialize the attenuator."""
        super().__init__(component_id, name)
        self.add_port("input", "input")
        self.add_port("output", "output")
        
        # Default attenuator parameters
        self.attenuation = 0.0  # dB
        self.max_power = 30.0  # dBm
        self.frequency_range = (0.0, 6e9)  # Hz
        self.variable = False  # Fixed or variable attenuator
        self.control_voltage = 0.0  # V (for voltage-controlled attenuators)
    
    def get_attenuation(self, frequency: float) -> float:
        """Get the attenuation at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Attenuation in dB (positive value)
        """
        # Check if frequency is within range
        min_freq, max_freq = self.frequency_range
        
        if frequency < min_freq or frequency > max_freq:
            # Add 1dB penalty for out-of-range frequencies
            return self.attenuation + 1.0
            
        return self.attenuation
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the attenuator."""
        # Calculate attenuation in linear scale
        attenuation_linear = 10 ** (-self.get_attenuation(frequency) / 20)
        
        # Simple 2-port S-parameter matrix for an attenuator
        s_params = np.zeros((2, 2), dtype=complex)
        s_params[0, 0] = 0.05  # S11: Input reflection
        s_params[0, 1] = attenuation_linear  # S12: Reverse transmission
        s_params[1, 0] = attenuation_linear  # S21: Forward transmission
        s_params[1, 1] = 0.05  # S22: Output reflection
        
        return s_params


class Switch(RFComponent):
    """RF switch component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Switch"):
        """Initialize the switch."""
        super().__init__(component_id, name)
        self.add_port("common", "bidirectional")
        self.add_port("port1", "bidirectional")
        self.add_port("port2", "bidirectional")
        
        # Default switch parameters
        self.insertion_loss = 0.5  # dB
        self.isolation = 30.0  # dB
        self.frequency_range = (0.0, 6e9)  # Hz
        self.active_port = "port1"  # Currently connected port
        self.switching_time = 1e-6  # seconds
    
    def set_active_port(self, port_name: str):
        """Set the active (connected) port.
        
        Args:
            port_name: Name of the port to connect to common
            
        Raises:
            ValueError: If port_name is not a valid port
        """
        if port_name not in self.ports or port_name == "common":
            raise ValueError(f"Invalid port name: {port_name}")
            
        self.active_port = port_name
    
    def get_insertion_loss(self, frequency: float) -> float:
        """Get the insertion loss at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Insertion loss in dB (positive value)
        """
        # Check if frequency is within range
        min_freq, max_freq = self.frequency_range
        
        if frequency < min_freq or frequency > max_freq:
            # Add 0.5dB penalty for out-of-range frequencies
            return self.insertion_loss + 0.5
            
        return self.insertion_loss
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the switch."""
        # Calculate insertion loss and isolation in linear scale
        loss_linear = 10 ** (-self.get_insertion_loss(frequency) / 20)
        isolation_linear = 10 ** (-self.isolation / 20)
        
        # 3-port S-parameter matrix for a switch
        s_params = np.zeros((3, 3), dtype=complex)
        
        # Reflection coefficients
        s_params[0, 0] = 0.1  # Common port reflection
        s_params[1, 1] = 0.1  # Port 1 reflection
        s_params[2, 2] = 0.1  # Port 2 reflection
        
        # Transmission coefficients
        if self.active_port == "port1":
            # Common to Port 1 transmission
            s_params[0, 1] = loss_linear
            s_params[1, 0] = loss_linear
            
            # Common to Port 2 isolation
            s_params[0, 2] = isolation_linear
            s_params[2, 0] = isolation_linear
            
            # Port 1 to Port 2 isolation
            s_params[1, 2] = isolation_linear
            s_params[2, 1] = isolation_linear
        else:
            # Common to Port 2 transmission
            s_params[0, 2] = loss_linear
            s_params[2, 0] = loss_linear
            
            # Common to Port 1 isolation
            s_params[0, 1] = isolation_linear
            s_params[1, 0] = isolation_linear
            
            # Port 1 to Port 2 isolation
            s_params[1, 2] = isolation_linear
            s_params[2, 1] = isolation_linear
        
        return s_params


class Circulator(RFComponent):
    """RF circulator component."""
    
    def __init__(self, component_id: Optional[str] = None, name: str = "Circulator"):
        """Initialize the circulator."""
        super().__init__(component_id, name)
        self.add_port("port1", "bidirectional")
        self.add_port("port2", "bidirectional")
        self.add_port("port3", "bidirectional")
        
        # Default circulator parameters
        self.insertion_loss = 0.3  # dB
        self.isolation = 20.0  # dB
        self.frequency_range = (0.0, 6e9)  # Hz
        self.vswr = 1.2  # Voltage Standing Wave Ratio
    
    def get_insertion_loss(self, frequency: float) -> float:
        """Get the insertion loss at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Insertion loss in dB (positive value)
        """
        # Check if frequency is within range
        min_freq, max_freq = self.frequency_range
        
        if frequency < min_freq or frequency > max_freq:
            # Add 1dB penalty for out-of-range frequencies
            return self.insertion_loss + 1.0
            
        return self.insertion_loss
    
    def get_s_parameters(self, frequency: float) -> np.ndarray:
        """Get the S-parameters of the circulator."""
        # Calculate insertion loss and isolation in linear scale
        loss_linear = 10 ** (-self.get_insertion_loss(frequency) / 20)
        isolation_linear = 10 ** (-self.isolation / 20)
        
        # Calculate reflection coefficient from VSWR
        reflection = (self.vswr - 1) / (self.vswr + 1)
        
        # 3-port S-parameter matrix for a circulator
        s_params = np.zeros((3, 3), dtype=complex)
        
        # Reflection coefficients
        s_params[0, 0] = reflection  # Port 1 reflection
        s_params[1, 1] = reflection  # Port 2 reflection
        s_params[2, 2] = reflection  # Port 3 reflection
        
        # Transmission coefficients (clockwise circulation)
        # Port 1 -> Port 2
        s_params[1, 0] = loss_linear
        # Port 2 -> Port 3
        s_params[2, 1] = loss_linear
        # Port 3 -> Port 1
        s_params[0, 2] = loss_linear
        
        # Isolation coefficients (counter-clockwise)
        # Port 2 -> Port 1
        s_params[0, 1] = isolation_linear
        # Port 3 -> Port 2
        s_params[1, 2] = isolation_linear
        # Port 1 -> Port 3
        s_params[2, 0] = isolation_linear
        
        return s_params