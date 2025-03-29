"""Optimized simulation engine for high-performance RF simulation."""

import heapq
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Optional, Dict, Set
# Import both ThreadPoolExecutor and ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from helios.core.data_structures import Scenario, Platform, Signal, Position
from helios.core.interfaces import IDataRecorder, IPropagationModel, ISimulationEngine # Import ISimulationEngine
from helios.utils.logger import get_logger
from helios.environment.clutter import DiscreteClutterModel

# Type alias for event callbacks
EventCallback = Callable[['OptimizedSimulationEngine', float, Any], None]

# Add to imports at the top
from helios.environment.hpm_coupling import HPMCouplingModel, HPMEffect

@dataclass(order=True)
class Event:
    """Represents a scheduled event in the simulation."""
    time: float
    priority: int  # Lower number means higher priority for same-time events
    type: str = field(compare=False)
    data: Optional[Any] = field(default=None, compare=False)
    callback: Optional[EventCallback] = field(default=None, compare=False)

# Inherit from ISimulationEngine
class OptimizedSimulationEngine(ISimulationEngine):
    """
    High-performance simulation engine optimized for low-latency processing.
    Uses parallel processing for computationally intensive tasks.
    Implements the ISimulationEngine interface.
    """

    # Update the OptimizedSimulationEngine __init__ method to include HPM coupling
    def __init__(self,
                 scenario: Scenario, # Keep scenario for initialization
                 propagation_model: IPropagationModel,
                 clutter_model: Optional[DiscreteClutterModel] = None,
                 hpm_coupling_model: Optional[HPMCouplingModel] = None,  # Add this parameter
                 log_level: int = logging.INFO,
                 data_recorder: Optional[IDataRecorder] = None,
                 max_workers: int = 4):
        """
        Initialize the optimized simulation engine.
        """
        self.logger = get_logger(__name__)
        self.logger.setLevel(log_level)

        # Models and components
        self.propagation_model = propagation_model
        self.clutter_model = clutter_model
        self.hpm_coupling_model = hpm_coupling_model
        self.data_recorder = data_recorder
        self.max_workers = max_workers

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Performance tracking
        self.processing_times = {
            'propagation': [],
            'platform_update': [],
            'signal_processing': [],
            'event_processing': []
        }

        # Spatial acceleration structure (simple grid-based)
        self.spatial_grid = {}
        self.grid_cell_size = 100.0  # meters

        # Event types (can be extended)
        self._EVENT_PLATFORM_UPDATE = "PLATFORM_UPDATE"
        self._EVENT_SIGNAL_PROPAGATION = "SIGNAL_PROPAGATION"
        self._EVENT_SIMULATION_END = "SIMULATION_END"

        self._running = False # Flag to indicate if simulation is running

        # Call initialize to set up scenario-specific state
        self.initialize(scenario)

    # Implement the initialize method from ISimulationEngine
    def initialize(self, scenario: Scenario) -> None:
        """Initialize the simulation with a scenario."""
        # Stop executor if it's running from a previous simulation
        if hasattr(self, 'executor') and self.executor:
             # Allow current tasks to complete before shutting down
            self.executor.shutdown(wait=True)
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.scenario = scenario
        self._current_time = scenario.start_time
        self._event_queue = []
        heapq.heapify(self._event_queue)
        self._running = False # Reset running state

        # Reset spatial grid and performance times
        self.spatial_grid = {}
        self.processing_times = { k: [] for k in self.processing_times }

        # Initialize spatial grid with platform starting positions
        for platform_id, platform in self.scenario.platforms.items():
            cell = self._get_grid_cell(platform.position)
            if cell not in self.spatial_grid:
                self.spatial_grid[cell] = set()
            self.spatial_grid[cell].add(platform_id)

        self._schedule_initial_events()
        self.logger.info(f"Optimized Engine initialized for scenario '{scenario.name}'.")


    # Implement the current_time property from ISimulationEngine
    @property
    def current_time(self) -> float:
        """Get the current simulation time."""
        return self._current_time

    def _get_grid_cell(self, position: Position) -> Tuple[int, int, int]:
         """Calculate the grid cell coordinates for a given position."""
         x_cell = int(position.x // self.grid_cell_size)
         y_cell = int(position.y // self.grid_cell_size)
         z_cell = int(position.z // self.grid_cell_size)
         return (x_cell, y_cell, z_cell)

    def _update_platform_grid_position(self, platform_id: str, old_pos: Position, new_pos: Position):
        """Update platform position in spatial grid."""
        old_cell = self._get_grid_cell(old_pos)
        new_cell = self._get_grid_cell(new_pos)
        
        if old_cell != new_cell:
            # Remove from old cell
            if old_cell in self.spatial_grid and platform_id in self.spatial_grid[old_cell]:
                self.spatial_grid[old_cell].remove(platform_id)
            
            # Add to new cell
            if new_cell not in self.spatial_grid:
                self.spatial_grid[new_cell] = set()
            self.spatial_grid[new_cell].add(platform_id)
    
    def _get_nearby_platforms(self, position: Position, radius: float) -> Set[str]:
        """Get platforms within radius of position using spatial grid."""
        center_cell = self._get_grid_cell(position)
        cell_radius = int(radius / self.grid_cell_size) + 1
        
        nearby_platforms = set()
        
        # Check cells in a cube around the center cell
        for x_offset in range(-cell_radius, cell_radius + 1):
            for y_offset in range(-cell_radius, cell_radius + 1):
                for z_offset in range(-cell_radius, cell_radius + 1):
                    cell = (center_cell[0] + x_offset, 
                           center_cell[1] + y_offset, 
                           center_cell[2] + z_offset)
                    
                    if cell in self.spatial_grid:
                        nearby_platforms.update(self.spatial_grid[cell])
        
        return nearby_platforms
    
    def _schedule_initial_events(self):
        """Schedule initial events based on the scenario."""
        # Schedule platform updates
        update_interval = 0.01  # 10ms default update interval
        for platform_id in self.scenario.platforms:
            self.schedule_event(
                self.current_time + update_interval,
                self._EVENT_PLATFORM_UPDATE,
                platform_id
            )
        
        # Schedule initial signals
        for signal in self.scenario.initial_signals:
            self.schedule_event(
                signal.emission_time,
                self._EVENT_SIGNAL_PROPAGATION,
                signal
            )
        
        # Schedule simulation end
        self.schedule_event(
            self.scenario.start_time + self.scenario.duration,
            self._EVENT_SIMULATION_END
        )
    
    # Align schedule_event signature with ISimulationEngine
    def schedule_event(self, time: float, event_type: str, data: Any = None,
                      callback: Optional[EventCallback] = None, priority: int = 0):
        """Schedule an event to occur at a specific time."""
        if time < self._current_time:
            self.logger.warning(f"Attempted to schedule event at time {time} which is before current time {self._current_time}. Skipping.")
            return
        event = Event(time=time, priority=priority, type=event_type, data=data, callback=callback)
        heapq.heappush(self._event_queue, event)
    
    # Align run signature with ISimulationEngine
    def run(self, duration: Optional[float] = None) -> None: # Return type is None as per interface
        """Run the simulation for specified duration."""
        if self._running:
            self.logger.warning("Simulation is already running.")
            return

        if duration is None:
            # Calculate end_time based on scenario duration if not provided
            end_time = self.scenario.start_time + self.scenario.duration
        else:
            end_time = self._current_time + duration

        self.logger.info(f"Starting simulation run from t={self._current_time:.6f}s to t={end_time:.6f}s")

        self._running = True
        start_wall_time = time.time()
        event_count = 0

        try:
            # Main simulation loop
            while self._event_queue and self._event_queue[0].time <= end_time:
                # Get the next event
                event = heapq.heappop(self._event_queue)

                # Advance time to the event time
                self._current_time = event.time

                # Update HPM effects (if model exists)
                if self.hpm_coupling_model:
                    self.hpm_coupling_model.update_effects(self._current_time, self.scenario.platforms)

                # Process the event
                event_start = time.time()
                self._process_event(event)
                event_duration = time.time() - event_start

                # Record processing time
                self.processing_times['event_processing'].append(event_duration)
                event_count += 1

                # Periodically log progress (example: every 1000 events)
                if event_count % 1000 == 0:
                    elapsed = time.time() - start_wall_time
                    self.logger.info(f"Processed {event_count} events, "
                                     f"sim time: {self._current_time:.6f}s, "
                                     f"wall time: {elapsed:.3f}s")

            # Check if simulation ended because the queue became empty before end_time
            if not self._event_queue and self._current_time < end_time:
                 self.logger.info(f"Simulation ended early at t={self._current_time:.6f}s due to empty event queue.")
                 self._current_time = end_time # Advance time to the requested end time

            # Ensure simulation time reaches the intended end_time if duration was specified
            elif self._current_time < end_time:
                  self._current_time = end_time

        except Exception as e:
            self.logger.error(f"Error during simulation run: {e}", exc_info=True)
            # Optionally re-raise or handle specific exceptions
            raise
        finally:
            self._running = False
            # Shutdown the executor cleanly
            self.executor.shutdown(wait=True)
            # Recreate executor for potential future runs or steps
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            # Log final performance statistics
            total_wall_time = time.time() - start_wall_time
            self.logger.info(f"Simulation run finished at t={self._current_time:.6f}s. Processed {event_count} events in {total_wall_time:.3f}s wall time.")
            self._log_performance_summary()


    def _log_performance_summary(self):
        """Log summary statistics of processing times."""
        # ... Implementation to calculate and log avg/max/min times ...
        pass

    # Implement the step method from ISimulationEngine
    def step(self, time_step: float) -> None:
        """Advance the simulation by a single time step."""
        if self._running:
            self.logger.warning("Cannot step while simulation is running via run().")
            return
        if time_step <= 0:
            self.logger.warning("Time step must be positive.")
            return

        end_time = self._current_time + time_step
        self.logger.debug(f"Stepping simulation from t={self._current_time:.6f}s to t={end_time:.6f}s")

        event_count = 0
        step_start_wall = time.time()
        try:
             self._running = True
             while self._event_queue and self._event_queue[0].time <= end_time:
                event = heapq.heappop(self._event_queue)
                self._current_time = event.time # Advance time to event

                 # Update HPM effects
                if self.hpm_coupling_model:
                    self.hpm_coupling_model.update_effects(self._current_time, self.scenario.platforms)

                self._process_event(event)
                event_count += 1

             # Advance time to the end of the step if no more events occurred within the step
             self._current_time = end_time

        except Exception as e:
             self.logger.error(f"Error during simulation step: {e}", exc_info=True)
             raise # Re-raise after logging
        finally:
            step_wall_time = time.time() - step_start_wall
            self.logger.debug(f"Step completed. Processed {event_count} events in {step_wall_time:.6f}s wall time.")
            self._running = False


    def _process_event(self, event: Event):
        """Process a single simulation event."""
        self.logger.debug(f"Processing event: {event.type} at time {event.time}")
        if event.type == self._EVENT_PLATFORM_UPDATE:
            if isinstance(event.data, str):
                self._handle_platform_update(event.data)
            else:
                self.logger.warning(f"Invalid platform_id type: {type(event.data)}")
        elif event.type == self._EVENT_SIGNAL_PROPAGATION:
            if isinstance(event.data, Signal):
                self._handle_signal_propagation(event.data)
            else:
                self.logger.warning(f"Invalid signal data type: {type(event.data)}")
        elif event.type == self._EVENT_SIMULATION_END:
            self.logger.info("Simulation end event reached")
        else:
            # Custom event with callback
            if event.callback:
                event.callback(self, self.current_time, event.data)
            else:
                self.logger.warning(f"Unknown event type: {event.type}")
    
    def _handle_platform_update(self, platform_id: str):
        """Handle platform update event."""
        start_time = time.time()
        
        if platform_id not in self.scenario.platforms:
            return
        
        platform = self.scenario.platforms[platform_id]
        old_position = Position(platform.position.x, platform.position.y, platform.position.z)
        
        # Update platform position based on velocity
        platform.update_position(0.01)  # 10ms update
        
        # Update spatial grid
        self._update_platform_grid_position(platform_id, old_position, platform.position)
        
        # Schedule next update
        self.schedule_event(
            self.current_time + 0.01,  # 10ms update interval
            self._EVENT_PLATFORM_UPDATE,
            platform_id
        )
        
        # Record processing time
        self.processing_times['platform_update'].append(time.time() - start_time)
    
    def _handle_signal_propagation(self, signal: Signal):
        """Handle signal propagation event using parallel processing."""
        start_time = time.time()
        
        # Find potential receivers (platforms within range)
        # Use spatial acceleration structure to find nearby platforms
        max_range = 100000  # 100km max range (can be adjusted based on signal power)
        nearby_platform_ids = self._get_nearby_platforms(signal.origin, max_range)
        
        # Process signal propagation in parallel
        if len(nearby_platform_ids) > 1 and self.max_workers > 1:
            # Parallel processing for multiple receivers
            futures = []
            for platform_id in nearby_platform_ids:
                if platform_id in self.scenario.platforms:
                    platform = self.scenario.platforms[platform_id]
                    futures.append(
                        self.executor.submit(
                            self._process_signal_at_platform,
                            signal, platform
                        )
                    )
            
            # Collect results
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in parallel signal processing: {e}")
        else:
            # Sequential processing for single receiver or small number
            for platform_id in nearby_platform_ids:
                if platform_id in self.scenario.platforms:
                    platform = self.scenario.platforms[platform_id]
                    self._process_signal_at_platform(signal, platform)
        
        # Process clutter reflections if clutter model exists
        if self.clutter_model:
            # For each nearby platform, generate reflections
            for platform_id in nearby_platform_ids:
                if platform_id in self.scenario.platforms:
                    platform = self.scenario.platforms[platform_id]
                    reflected_signals = self.clutter_model.generate_reflections(signal, platform)
                    
                    # Schedule propagation of reflected signals
                    for reflected in reflected_signals:
                        self.schedule_event(
                            reflected.emission_time,
                            self._EVENT_SIGNAL_PROPAGATION,
                            reflected
                        )
        
        # Record processing time
        prop_time = time.time() - start_time
        self.processing_times['propagation'].append(prop_time)
    
    # Update the _process_signal_at_platform method to include HPM coupling
    def _process_signal_at_platform(self, signal: Signal, platform: Platform) -> None:
        """Process a signal at a specific platform."""
        start_time = time.time()
        
        # Skip if signal originated from this platform
        if signal.source_id == platform.id:
            return
        
        # Apply propagation model
        modified_signal = self.propagation_model.apply_propagation_effects(
            signal, platform, self.scenario.environment
        )
        
        # Check if signal is above noise floor
        noise_power = self.scenario.environment.calculate_noise_power(
            signal.waveform.bandwidth
        )
        
        if modified_signal.power > noise_power:
            # Add to platform's received signals
            platform.received_signals[modified_signal.id] = modified_signal
            
            # Update combined signal power (simplified)
            if platform.combined_signal_power == float('-inf'):
                platform.combined_signal_power = modified_signal.power
            else:
                # Power addition in dB domain (approximate)
                max_power = max(platform.combined_signal_power, modified_signal.power)
                power_diff = abs(platform.combined_signal_power - modified_signal.power)
                if power_diff > 10:
                    # If difference is large, just use the max
                    platform.combined_signal_power = max_power
                else:
                    # Otherwise, approximate power addition
                    platform.combined_signal_power = max_power + 10 * np.log10(1 + 10**(-power_diff/10))
            
            # Record data if recorder exists
            if self.data_recorder:
                # Change from record_signal_reception to record_signal
                self.data_recorder.record_signal(
                    self.current_time, modified_signal, platform.id
                )
        
        # Record processing time
        self.processing_times['signal_processing'].append(time.time() - start_time)

    def __del__(self):
        """Ensure executor is shut down when the engine is deleted."""
        if hasattr(self, 'executor') and self.executor:
            # Don't wait indefinitely here, as __del__ can block garbage collection
            self.executor.shutdown(wait=False)
