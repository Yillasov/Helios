"""Real-time data streaming between simulation and hardware."""

import time
import queue
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging

from helios.hardware.interfaces import IRadioHardwareInterface
from helios.core.data_structures import Signal

logger = logging.getLogger(__name__)

class StreamingBuffer:
    """Thread-safe buffer for streaming data between simulation and hardware."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the streaming buffer."""
        self.queue = queue.Queue(maxsize=max_size)
        self.closed = False
    
    def put(self, data: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add data to the buffer."""
        if self.closed:
            return False
        try:
            self.queue.put(data, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Get data from the buffer."""
        if self.closed and self.queue.empty():
            return None
        try:
            return self.queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self):
        """Close the buffer."""
        self.closed = True


class HardwareStreamer:
    """Manages real-time data streaming between simulation and hardware."""
    
    def __init__(self, device: IRadioHardwareInterface):
        """Initialize the hardware streamer."""
        self.device = device
        self.sim_to_hw_buffer = StreamingBuffer()
        self.hw_to_sim_buffer = StreamingBuffer()
        self.running = False
        self.hw_thread = None
        self.sim_thread = None
        self.sample_rate = 2e6  # Default 2 MHz
        self.center_freq = 915e6  # Default 915 MHz
        self.gain = 30  # Default gain
        
        # Callbacks
        self.on_hw_samples = None
        self.on_sim_signal = None
    
    def start_streaming(self, 
                       sample_rate: float = 2e6, 
                       center_freq: float = 915e6,
                       gain: float = 30) -> bool:
        """Start the streaming threads."""
        if self.running:
            logger.warning("Streaming already running")
            return False
        
        # Get device status to check initialization
        status = self.device.get_hardware_status()
        if "error" in status and "not initialized" in status["error"].lower():
            logger.error("Device not initialized")
            return False
        
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        
        # Configure device
        self.device.set_rx_parameters({
            "frequency": center_freq,
            "sample_rate": sample_rate,
            "gain": gain
        })
        
        self.running = True
        
        # Start hardware thread (receiving samples from hardware)
        self.hw_thread = threading.Thread(
            target=self._hardware_loop,
            daemon=True
        )
        self.hw_thread.start()
        
        # Start simulation thread (sending samples to hardware)
        self.sim_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.sim_thread.start()
        
        logger.info(f"Started streaming at {center_freq/1e6:.2f} MHz, {sample_rate/1e6:.2f} Msps")
        return True
    
    def stop_streaming(self):
        """Stop the streaming threads."""
        if not self.running:
            return
        
        self.running = False
        self.sim_to_hw_buffer.close()
        self.hw_to_sim_buffer.close()
        
        # Wait for threads to finish
        if self.hw_thread:
            self.hw_thread.join(timeout=1.0)
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)
        
        logger.info("Stopped streaming")
    
    def _hardware_loop(self):
        """Hardware thread: receives samples from hardware and puts them in buffer."""
        logger.debug("Hardware thread started")
        
        while self.running:
            try:
                # Receive samples from hardware
                samples, metadata = self.device.receive_samples(
                    10000,  # 10k samples per batch
                    self.center_freq,
                    self.sample_rate,
                    self.gain
                )
                
                # Add timestamp
                metadata["timestamp"] = time.time()
                
                # Put in buffer
                data_packet = {"samples": samples, "metadata": metadata}
                self.hw_to_sim_buffer.put(data_packet, block=False)
                
                # Call callback if registered
                if self.on_hw_samples:
                    self.on_hw_samples(samples, metadata)
                
            except Exception as e:
                logger.error(f"Error in hardware loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _simulation_loop(self):
        """Simulation thread: gets samples from buffer and sends to hardware."""
        logger.debug("Simulation thread started")
        
        while self.running:
            try:
                # Get data from buffer
                data = self.sim_to_hw_buffer.get(timeout=0.1)
                if data is None:
                    continue
                
                # Check if it's a Signal object or raw samples
                if isinstance(data, Signal):
                    # Convert Signal to samples
                    samples = self._signal_to_samples(data)
                    
                    # Call callback if registered
                    if self.on_sim_signal:
                        self.on_sim_signal(data)
                else:
                    # Assume it's already samples
                    samples = data
                
                # Transmit samples
                self.device.transmit_samples(
                    samples,
                    self.center_freq,
                    self.sample_rate,
                    self.gain
                )
                
            except queue.Empty:
                pass  # No data available
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _signal_to_samples(self, signal: Signal) -> np.ndarray:
        """Convert a Signal object to IQ samples."""
        # This is a simplified conversion - in a real implementation,
        # you would use the signal's modulation, frequency, etc.
        
        # Fix: Access duration from the waveform property instead of directly from signal
        duration = signal.waveform.duration
        
        # Handle case where duration might be None
        if duration is None or duration <= 0:
            duration = 0.1  # Default to 100ms if duration is not set
            
        num_samples = int(duration * self.sample_rate)
        
        # Generate simple samples (e.g., sine wave)
        t = np.arange(num_samples) / self.sample_rate
        samples = 0.5 * np.exp(2j * np.pi * 1000 * t)  # 1 kHz tone
        
        return samples
    
    def send_to_hardware(self, data: Any) -> bool:
        """Send data to hardware (can be Signal or raw samples)."""
        return self.sim_to_hw_buffer.put(data, block=False)
    
    def get_from_hardware(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get latest samples from hardware."""
        return self.hw_to_sim_buffer.get(block=block, timeout=timeout)
    
    def register_hw_callback(self, callback: Callable[[np.ndarray, Dict[str, Any]], None]):
        """Register callback for hardware samples."""
        self.on_hw_samples = callback
    
    def register_sim_callback(self, callback: Callable[[Signal], None]):
        """Register callback for simulation signals."""
        self.on_sim_signal = callback


class SimulationHardwareBridge:
    """Bridge between simulation engine and hardware devices."""
    
    def __init__(self):
        """Initialize the bridge."""
        self.streamers = {}  # device_id -> HardwareStreamer
        self.sim_time = 0.0
        self.time_scale = 1.0  # Simulation time to real time ratio
    
    def add_device(self, device_id: str, device: IRadioHardwareInterface) -> bool:
        """Add a hardware device to the bridge."""
        if device_id in self.streamers:
            logger.warning(f"Device {device_id} already added")
            return False
        
        self.streamers[device_id] = HardwareStreamer(device)
        return True
    
    def start_streaming(self, device_id: str, **kwargs) -> bool:
        """Start streaming for a specific device."""
        if device_id not in self.streamers:
            logger.error(f"Device {device_id} not found")
            return False
        
        return self.streamers[device_id].start_streaming(**kwargs)
    
    def stop_streaming(self, device_id: str) -> bool:
        """Stop streaming for a specific device."""
        if device_id not in self.streamers:
            logger.error(f"Device {device_id} not found")
            return False
        
        self.streamers[device_id].stop_streaming()
        return True
    
    def update_sim_time(self, sim_time: float):
        """Update the current simulation time."""
        self.sim_time = sim_time
    
    def send_signal_to_hardware(self, device_id: str, signal: Signal) -> bool:
        """Send a simulation signal to hardware."""
        if device_id not in self.streamers:
            logger.error(f"Device {device_id} not found")
            return False
        
        return self.streamers[device_id].send_to_hardware(signal)
    
    def get_samples_from_hardware(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get latest samples from hardware."""
        if device_id not in self.streamers:
            logger.error(f"Device {device_id} not found")
            return None
        
        return self.streamers[device_id].get_from_hardware(block=False)
    
    def close(self):
        """Close all streamers."""
        for device_id, streamer in self.streamers.items():
            streamer.stop_streaming()


"""Simplified bidirectional data streaming."""

import queue
import threading
from typing import Optional, Any

class DataBridge:
    """Manages bidirectional data flow between sim and hardware."""
    
    def __init__(self):
        self.sim_to_hw = queue.Queue()  # Simulation → Hardware
        self.hw_to_sim = queue.Queue()  # Hardware → Simulation
        self.running = False
        
    def start(self):
        """Start data flow threads."""
        self.running = True
        threading.Thread(target=self._hw_to_sim_loop, daemon=True).start()
        threading.Thread(target=self._sim_to_hw_loop, daemon=True).start()
        
    def stop(self):
        """Stop data flow."""
        self.running = False
        
    def _hw_to_sim_loop(self):
        """Hardware → Simulation data flow."""
        while self.running:
            # Get data from hardware
            hw_data = self._read_hardware()
            if hw_data:
                self.hw_to_sim.put(hw_data)
                
    def _sim_to_hw_loop(self):
        """Simulation → Hardware data flow."""
        while self.running:
            # Get data from simulation
            sim_data = self.sim_to_hw.get()
            if sim_data:
                self._write_hardware(sim_data)
                
    def _read_hardware(self) -> Optional[Any]:
        """Read data from hardware (implement per device)."""
        pass
        
    def _write_hardware(self, data: Any):
        """Write data to hardware (implement per device)."""
        pass