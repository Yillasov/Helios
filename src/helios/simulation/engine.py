import heapq
import numpy as np 
import logging # Import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple, Optional

from helios.core.data_structures import Scenario, Platform, Signal # Assuming these are defined
from helios.core.interfaces import IDataRecorder # Import the interface
from helios.utils.logger import setup_logging, get_logger # Import logger utilities

# Type alias for event callbacks
EventCallback = Callable[['SimulationEngine', float, Any], None]

@dataclass(order=True)
class Event:
    """Represents a scheduled event in the simulation."""
    time: float
    priority: int # Lower number means higher priority for same-time events
    type: str = field(compare=False)
    data: Optional[Any] = field(default=None, compare=False)
    callback: Optional[EventCallback] = field(default=None, compare=False)

class SimulationEngine:
    """Manages the simulation execution, timing, and event scheduling."""

    def __init__(self,
                 scenario: Scenario,
                 log_level: int = logging.INFO,
                 data_recorder: Optional[IDataRecorder] = None): # Add optional data_recorder parameter
        """Initialize the simulation engine with a scenario.

        Args:
            scenario: The scenario to simulate
            log_level: The minimum logging level for this simulation run.
            data_recorder: Optional component for recording simulation results.
        """
        # Configure logging for this engine instance run
        # Alternatively, setup_logging can be called once at application startup
        setup_logging(level=log_level)
        self.logger = get_logger(__name__) # Get a logger for this module

        self.scenario: Scenario = scenario
        self.current_time: float = scenario.start_time
        self._event_queue: List[Event] = []
        heapq.heapify(self._event_queue) # Initialize the heap
        self.data_recorder: Optional[IDataRecorder] = data_recorder # Store the data recorder instance

        # --- Example Internal Event Types ---
        # These could be expanded significantly
        self._INTERNAL_EVENT_PLATFORM_UPDATE = "PLATFORM_UPDATE"
        self._INTERNAL_EVENT_SIMULATION_END = "SIMULATION_END"

        self.logger.info(f"Simulation Engine initialized for scenario '{scenario.name}' (ID: {scenario.id}).")
        # Schedule initial events based on the scenario
        self._schedule_initial_events()

    def _schedule_initial_events(self) -> None:
        """Schedules the first events necessary to start the simulation."""
        self.logger.debug("Scheduling initial events...")

        # --- Platform Update Scheduling ---
        # Get frequency from config (assuming it's loaded into scenario or accessible)
        # For now, let's assume a default or read directly if possible.
        # It's better practice to have the config loaded into the Scenario object.
        # Assuming self.scenario has access to config or relevant parameters.
        # Example: self.scenario.config['simulation']['platform_update_frequency']
        platform_update_frequency = getattr(self.scenario, 'platform_update_frequency', 1.0) # Default 1 Hz
        if platform_update_frequency > 0:
            self.platform_update_interval = 1.0 / platform_update_frequency
            # Schedule the first platform update event at the start time (or slightly after if needed)
            # Using a low priority (higher number) so other t=0 events can run first if needed
            self.schedule_event(self.current_time, 10, self._INTERNAL_EVENT_PLATFORM_UPDATE, callback=lambda engine, time, data: self._handle_platform_update(time, data))
            self.logger.debug(f"Scheduled initial {self._INTERNAL_EVENT_PLATFORM_UPDATE} at time {self.current_time} with interval {self.platform_update_interval:.4f}s")
        else:
            self.platform_update_interval = float('inf') # Effectively disable updates if freq <= 0
            self.logger.debug("Platform update frequency is zero or negative, periodic updates disabled.")

        # --- Simulation End Scheduling ---
        end_time = self.scenario.start_time + self.scenario.duration
        self.schedule_event(end_time, 0, self._INTERNAL_EVENT_SIMULATION_END) # High priority (0)
        self.logger.debug(f"Scheduled {self._INTERNAL_EVENT_SIMULATION_END} at time {end_time}")

        # --- Initial Signal Scheduling ---
        for signal in self.scenario.initial_signals:
            if signal.emission_time >= self.current_time:
                self.schedule_event(
                    signal.emission_time, 
                    5, 
                    "SIGNAL_START", 
                    data=signal, 
                    callback=lambda engine, time, data: self._handle_signal_start(time, data)
                )
                self.logger.debug(f"Initial signal {signal.id} emission scheduled at {signal.emission_time}")

    def schedule_event(self, time: float, priority: int, event_type: str, data: Optional[Any] = None, callback: Optional[EventCallback] = None) -> None:
        """Schedule a new event to occur at a specific time.

        Args:
            time: Simulation time for the event. Must be >= current_time.
            priority: Priority for events at the same time (lower number is higher priority).
            event_type: A string identifying the type of event.
            data: Optional data associated with the event.
            callback: Optional function to call when the event is processed.
                      Callback signature: callback(engine, event_time, event_data)
        """
        if time < self.current_time:
            # Using warning log level instead of print
            self.logger.warning(f"Attempted to schedule event in the past (current: {self.current_time}, scheduled: {time}). Skipping.")
            return

        event = Event(time=time, priority=priority, type=event_type, data=data, callback=callback)
        heapq.heappush(self._event_queue, event)
        self.logger.debug(f"Scheduled event: Time={time}, Priority={priority}, Type={event_type}")

    def _process_next_event(self) -> Optional[Event]:
        """Processes the next event in the queue if its time has come.

        Returns:
            The processed event, or None if no event was ready.
        """
        if not self._event_queue:
            self.logger.debug("Event queue is empty.")
            return None # No events left

        next_event = self._event_queue[0] # Peek at the next event

        if next_event.time <= self.current_time:
            event = heapq.heappop(self._event_queue)
            # Update time ONLY if advancing time (can process multiple events at the same timestamp)
            # Allow processing multiple events at the *exact* same time
            if event.time > self.current_time:
                 self.logger.debug(f"Advancing time from {self.current_time} to {event.time}")
                 self.current_time = event.time # Correctly advances time only when needed

            self.logger.debug(f"Processing event: Time={event.time}, Priority={event.priority}, Type={event.type}")

            # Execute the callback if one is provided
            if event.callback:
                try:
                    event.callback(self, event.time, event.data)
                except Exception as e:
                    self.logger.error(f"Error executing callback for event {event.type} at time {event.time}: {e}", exc_info=True)
                    # Decide if simulation should stop on error, or just log and continue
            else:
                # Default logging if no specific callback
                self.logger.info(f"Processed unhandled event: Time={event.time}, Type={event.type}, Data={event.data}")

            return event
        else:
            # Next event is in the future, advance time directly to it
            self.logger.debug(f"No events at current time {self.current_time}. Advancing time to next event at {next_event.time}")
            self.current_time = next_event.time
            # Now process it in the next iteration or call _process_next_event again
            return None # Indicate time was advanced but no event processed *in this call*


    def run(self) -> None:
        """Runs the simulation until the event queue is empty or an end condition is met."""
        self.logger.info(f"Simulation starting at time {self.current_time} for duration {self.scenario.duration}")

        while self._event_queue:
            next_event_peek = self._event_queue[0]

            # Check for simulation end condition
            if next_event_peek.type == self._INTERNAL_EVENT_SIMULATION_END:
                 # Process the end event if its time has come or passed
                 if next_event_peek.time <= self.current_time:
                     self.logger.info(f"Simulation end event reached at time {next_event_peek.time}. Stopping.")
                     # Optionally process the end event itself if it has a callback
                     if next_event_peek.callback:
                          event_to_process = heapq.heappop(self._event_queue) # Remove event first
                          try:
                               if event_to_process.callback:  # Check if callback exists before calling
                                   event_to_process.callback(self, event_to_process.time, event_to_process.data)
                          except Exception as e:
                               self.logger.error(f"Error executing callback for end event: {e}", exc_info=True)
                     break # Exit the loop
                 # If end event time hasn't been reached yet, continue processing other events

            # Process the next event whose time has come, or advance time to the next event
            processed_event = self._process_next_event()
            if processed_event is None and not self._event_queue:
                 # If _process_next_event returned None (meaning it advanced time)
                 # and the queue is now empty, we should break cleanly.
                 self.logger.debug("Advanced time and event queue is now empty.")
                 break

            # Loop continues:
            # - if processed_event is not None (an event was handled at current_time)
            # - if processed_event is None but queue is not empty (time was advanced)


        self.logger.info(f"Simulation finished at time {self.current_time}")


    # --- Example Event Handlers ---

    def _handle_platform_update(self, event_time: float, event_data: Any) -> None:
        """Handles the periodic update of platform positions."""

        # Calculate time delta since the last platform update
        last_update_time = getattr(self, '_last_platform_update_time', self.scenario.start_time)
        time_delta = event_time - last_update_time
        self._last_platform_update_time = event_time # Store current event time as the last update time

        if time_delta <= 1e-9: # Avoid zero or negative delta if events occur at the same time
             self.logger.debug(f"Skipping platform update at {event_time:.6f} due to zero time delta.")
        else:
            self.logger.debug(f"Executing platform update at {event_time:.6f}. Delta since last: {time_delta:.6f}s")
            for platform_id, platform in self.scenario.platforms.items():
                # Update platform dynamics based on the actual time elapsed
                old_pos = (platform.position.x, platform.position.y, platform.position.z)
                platform.update_position(time_delta) # Use the actual elapsed time
                new_pos = (platform.position.x, platform.position.y, platform.position.z)
                self.logger.debug(f"  Updated platform '{platform.name}' ({platform_id}): {old_pos} -> {new_pos}")

                # Record state AFTER update (if recorder exists)
                if hasattr(self, 'data_recorder') and self.data_recorder:
                     # Check if the recorder interface method exists
                     if hasattr(self.data_recorder, 'record_platform_state'):
                        try:
                            self.data_recorder.record_platform_state(event_time, platform)
                        except Exception as e:
                            self.logger.error(f"Error recording platform state for {platform.id}: {e}", exc_info=True)


        # Schedule the next platform update event based on the interval
        next_update_time = event_time + self.platform_update_interval
        simulation_end_time = self.scenario.start_time + self.scenario.duration

        # Schedule next update only if it's within the simulation duration
        # Use a small tolerance for floating point comparisons
        if next_update_time <= (simulation_end_time + 1e-9):
             self.logger.debug(f"Scheduling next platform update at {next_update_time:.6f}")
             self.schedule_event(next_update_time, 10, self._INTERNAL_EVENT_PLATFORM_UPDATE, callback=lambda engine, time, data: self._handle_platform_update(time, data))
        else:
             self.logger.debug(f"Next platform update time {next_update_time:.6f} exceeds duration ({simulation_end_time:.6f}). Not rescheduling.")


    def _handle_signal_start(self, event_time: float, event_data: Any) -> None:
        """Handles the start of signal emission."""
        signal = event_data  # The signal object from the event data
        self.logger.info(f"Signal {signal.id} starts emission at {event_time}")
        
        # Add to active signals list (create if doesn't exist)
        if not hasattr(self, '_active_signals'):
            self._active_signals = []
        self._active_signals.append(signal)
        
        # Schedule signal interactions with all platforms
        for platform_id, platform in self.scenario.platforms.items():
            # Skip if the platform is the source of this signal
            if platform_id == signal.source_id:
                continue
                
            # Schedule signal interaction event
            self.schedule_event(
                event_time, 
                5,  # Medium priority
                "SIGNAL_INTERACTION", 
                data={"signal": signal, "platform_id": platform_id},
                callback=lambda engine, time, data: self._handle_signal_interaction(time, data)
            )

    def _handle_signal_interaction(self, event_time: float, event_data: Any) -> None:
        """Handles the interaction of a signal with a platform."""
        signal = event_data["signal"]
        platform_id = event_data["platform_id"]
        platform = self.scenario.platforms.get(platform_id)
        
        if not platform:
            self.logger.warning(f"Platform {platform_id} not found for signal interaction")
            return
            
        # Get the propagation model (assuming it's stored in the scenario or engine)
        prop_model = getattr(self.scenario, 'propagation_model', None)
        if not prop_model:
            self.logger.warning("No propagation model available for signal interaction")
            return
            
        # Apply propagation effects to the signal
        env_params = getattr(self.scenario, 'environment_parameters', None)
        modified_signal = prop_model.apply_propagation_effects(signal, platform, env_params)
        
        # Add to platform's received signals (create if doesn't exist)
        if not hasattr(platform, 'received_signals'):
            platform.received_signals = {}
        
        # Store the received signal, keyed by signal ID
        platform.received_signals[signal.id] = modified_signal
        
        # Calculate combined signal at the receiver
        self._calculate_combined_signal(event_time, platform)
        
        # Record the signal reception if a data recorder exists
        if hasattr(self, 'data_recorder') and self.data_recorder:
            if hasattr(self.data_recorder, 'record_signal'):
                try:
                    self.data_recorder.record_signal(event_time, modified_signal, platform_id)
                except Exception as e:
                    self.logger.error(f"Error recording signal for {platform_id}: {e}", exc_info=True)

    def _calculate_combined_signal(self, event_time: float, platform: Platform) -> None:
        """Calculate the combined effect of all signals at a receiver."""
        if not platform.received_signals:  # Now we can check directly
            return  # No signals to combine
            
        # Simple power summation (in linear scale)
        total_power_linear = 0.0
        
        # For each signal, convert dBm to linear power and sum
        for signal_id, signal in platform.received_signals.items():
            # Convert dBm to linear power (mW)
            power_mw = 10 ** (signal.power / 10)
            total_power_linear += power_mw
        
        # Convert back to dBm
        total_power_dbm = 10 * np.log10(total_power_linear) if total_power_linear > 0 else float('-inf')
        
        # Store the combined power on the platform
        platform.combined_signal_power = total_power_dbm
        
        self.logger.debug(f"Combined signal power at {platform.name}: {total_power_dbm:.2f} dBm")
        
        # Here you would typically trigger receiver processing based on the combined signal
        # This could include detection, demodulation, etc.