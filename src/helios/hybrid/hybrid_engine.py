"""Hybrid simulation engine that combines simulated and real hardware components."""

import time
import logging
from typing import Dict, List, Optional, Set, Any

from helios.core.data_structures import Scenario, Signal, Platform
from helios.core.interfaces import ISimulationEngine, IPropagationModel
from helios.hardware.data_streaming import SimulationHardwareBridge
from helios.hardware.calibration_tools import CalibrationTool
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class HybridSimulationEngine:
    """Manages hybrid simulation with both virtual and real hardware components."""
    
    def __init__(
        self,
        simulation_engine: ISimulationEngine,
        hardware_bridge: SimulationHardwareBridge,
        scenario: Scenario,  # Add scenario as a direct parameter
        time_scale: float = 1.0
    ):
        """Initialize the hybrid simulation engine.
        
        Args:
            simulation_engine: The simulation engine for virtual components
            hardware_bridge: Bridge to real hardware devices
            scenario: The simulation scenario
            time_scale: Simulation time to real time ratio (1.0 = real-time)
        """
        self.sim_engine = simulation_engine
        self.hw_bridge = hardware_bridge
        self.scenario = scenario  # Store scenario directly
        self.time_scale = time_scale
        self.running = False
        self.hardware_platforms: Set[str] = set()
        self.calibration_tools: Dict[str, CalibrationTool] = {}
        
    def register_hardware_platform(self, platform_id: str, device_id: str) -> bool:
        """Register a platform as being represented by real hardware.
        
        Args:
            platform_id: ID of the platform in the simulation
            device_id: ID of the hardware device
            
        Returns:
            Success status
        """
        if platform_id not in self.scenario.platforms:  # Use our own scenario reference
            logger.error(f"Platform {platform_id} not found in scenario")
            return False
            
        self.hardware_platforms.add(platform_id)
        logger.info(f"Registered platform {platform_id} as hardware device {device_id}")
        return True
        
    def register_calibration(self, device_id: str, calibration_file: str) -> bool:
        """Register calibration data for a hardware device.
        
        Args:
            device_id: ID of the hardware device
            calibration_file: Path to calibration file
            
        Returns:
            Success status
        """
        try:
            cal_tool = CalibrationTool(device_id)
            cal_tool.load_calibration(calibration_file)
            self.calibration_tools[device_id] = cal_tool
            logger.info(f"Loaded calibration for device {device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def run(self, duration: Optional[float] = None) -> None:
        """Run the hybrid simulation.
        
        Args:
            duration: Duration to run in seconds, or None to use scenario duration
        """
        if self.running:
            logger.warning("Hybrid simulation already running")
            return
            
        self.running = True
        start_time = time.time()
        
        # Start hardware streaming for all devices
        for device_id in self.hw_bridge.streamers:
            self.hw_bridge.start_streaming(device_id)
        
        # Hook into simulation engine events
        self._setup_event_hooks()
        
        try:
            # Run simulation with small steps to allow hardware interaction
            step_size = 0.05  # 50ms steps
            remaining_duration = duration
            
            while self.running:
                # Calculate step duration
                if remaining_duration is not None:
                    if remaining_duration <= 0:
                        break
                    current_step = min(step_size, remaining_duration)
                    remaining_duration -= current_step
                else:
                    current_step = step_size
                
                # Step simulation
                self.sim_engine.step(current_step)
                
                # Update hardware bridge with current simulation time
                self.hw_bridge.update_sim_time(self.sim_engine.current_time)
                
                # Process hardware data
                self._process_hardware_data()
                
                # Sleep to maintain time scale if needed
                self._maintain_time_scale(start_time)
                
        except Exception as e:
            logger.error(f"Error in hybrid simulation: {e}", exc_info=True)
        finally:
            self.running = False
            # Stop hardware streaming
            for device_id in self.hw_bridge.streamers:
                self.hw_bridge.stop_streaming(device_id)
    
    def stop(self) -> None:
        """Stop the hybrid simulation."""
        self.running = False
    
    def _setup_event_hooks(self) -> None:
        """Set up hooks for simulation events."""
        # This would be implemented by registering callbacks with the simulation engine
        # For signal propagation, platform updates, etc.
        pass
    
    def _process_hardware_data(self) -> None:
        """Process data from hardware devices."""
        for device_id in self.hw_bridge.streamers:
            # Get latest samples from hardware
            data = self.hw_bridge.get_samples_from_hardware(device_id)
            if data:
                # Process samples (e.g., convert to Signal, apply to simulation)
                self._inject_hardware_data_to_simulation(device_id, data)
    
    def _inject_hardware_data_to_simulation(self, device_id: str, data: Dict[str, Any]) -> None:
        """Inject hardware data into the simulation.
        
        Args:
            device_id: ID of the hardware device
            data: Data from hardware
        """
        # Implementation would depend on the specific simulation engine
        # This is a placeholder for the actual implementation
        pass
    
    def _maintain_time_scale(self, start_time: float) -> None:
        """Maintain the specified time scale between simulation and real time.
        
        Args:
            start_time: Wall clock time when simulation started
        """
        if self.time_scale <= 0:
            return  # Run as fast as possible
            
        expected_time = start_time + (self.sim_engine.current_time / self.time_scale)
        current_time = time.time()
        
        if current_time < expected_time:
            # Sleep to maintain time scale
            time.sleep(expected_time - current_time)