"""RF-guided munition trajectory and guidance simulations."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import uuid
from enum import Enum, auto

from helios.armament.guidance import GuidanceSystem, GuidanceMode, TrajectoryCalculator
from helios.core.data_structures import Position, Velocity, Orientation
from helios.effects.electronic_vulnerability import HPMVulnerabilityModel, ComponentVulnerability
from helios.utils.logger import get_logger
from helios.armament.terrain_propagation import TacticalTerrainModel

logger = get_logger(__name__)

class SeekingMode(Enum):
    """RF seeker modes for guided munitions."""
    ACTIVE_RADAR = auto()    # Munition emits and receives radar signals
    SEMI_ACTIVE = auto()     # External illumination, munition receives only
    ANTI_RADIATION = auto()  # Homes on enemy radar emissions
    DUAL_MODE = auto()       # Combines multiple seeking methods
    GPS_AIDED = auto()       # GPS + RF terminal guidance


@dataclass
class RFSeeker:
    """RF seeker for guided munitions."""
    mode: SeekingMode
    center_frequency: float  # Hz
    bandwidth: float         # Hz
    max_range: float         # meters
    field_of_view: float     # degrees
    tracking_accuracy: float = 1.0  # meters CEP (Circular Error Probable)
    jamming_resistance: float = 0.5  # 0-1 scale, higher is better
    
    # For active seekers
    peak_power: Optional[float] = None  # Watts
    
    def can_track_target(self, 
                         target_position: Position, 
                         munition_position: Position,
                         munition_orientation: Orientation,
                         jamming_level: float = 0.0,
                         terrain_model: Optional[TacticalTerrainModel] = None) -> bool:
        """Determine if the seeker can track the target."""
        # Calculate distance to target
        distance = munition_position.distance_to(target_position)
        
        # Check if target is within range
        if distance > self.max_range:
            return False
            
        # Calculate angle to target (simplified)
        dx = target_position.x - munition_position.x
        dy = target_position.y - munition_position.y
        dz = target_position.z - munition_position.z
        
        # Convert to spherical coordinates (azimuth, elevation)
        azimuth = np.arctan2(dy, dx)
        elevation = np.arcsin(dz / distance)
        
        # Convert to degrees
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)
        
        # Check if target is within field of view (simplified)
        # This is a basic check that doesn't account for full orientation
        if abs(azimuth_deg - np.degrees(munition_orientation.yaw)) > self.field_of_view / 2:
            return False
            
        if abs(elevation_deg - np.degrees(munition_orientation.pitch)) > self.field_of_view / 2:
            return False
        
        # Apply terrain effects if terrain model is provided
        if terrain_model:
            terrain_effects = terrain_model.calculate_terrain_effects(
                munition_position, 
                target_position,
                self.center_frequency
            )
            
            # If LOS is blocked and we're not using an active radar that can detect non-LOS targets
            if terrain_effects["los_blocked"] and self.mode != SeekingMode.ACTIVE_RADAR:
                return False
                
            # Calculate effective signal loss due to terrain
            terrain_loss = terrain_effects["total_effect_db"]
            
            # Adjust effective range based on terrain loss
            # For every 6dB of loss, effective range is halved (simplified model)
            range_factor = 10 ** (-terrain_loss / 20)  # Convert dB to linear scale
            effective_range = self.max_range * range_factor
            
            if distance > effective_range:
                logger.debug(f"Target out of effective range due to terrain effects: {terrain_loss:.1f} dB loss")
                return False
            
        # Check jamming effectiveness
        effective_jamming = jamming_level * (1 - self.jamming_resistance)
        if effective_jamming > 0.8:  # Arbitrary threshold
            return False
            
        return True


class RFGuidedMunition:
    """Simulation model for RF-guided munitions."""
    
    def __init__(self, 
                 seeker: RFSeeker,
                 guidance_mode: GuidanceMode = GuidanceMode.SEMI_ACTIVE):
        """Initialize the RF-guided munition simulation.
        
        Args:
            seeker: RF seeker configuration
            guidance_mode: Guidance system mode
        """
        self.id = str(uuid.uuid4())
        self.seeker = seeker
        self.guidance_system = GuidanceSystem(mode=guidance_mode)
        self.trajectory_calculator = TrajectoryCalculator()
        
        # Physical properties
        self.mass = 100.0  # kg
        self.drag_coefficient = 0.1
        self.cross_section = 0.05  # m²
        
        # State variables
        self.position = Position(0, 0, 0)
        self.velocity = Velocity(0, 0, 0)
        self.orientation = Orientation(0, 0, 0)
        self.active = True
        self.target_locked = False
        
        # Vulnerability model for electronic components
        self.vulnerability_model = None
        
    def initialize_vulnerability_model(self, model: HPMVulnerabilityModel):
        """Initialize the vulnerability model for this munition."""
        self.vulnerability_model = model
        
    def launch(self, 
              initial_position: Position,
              initial_velocity: Velocity,
              initial_orientation: Orientation):
        """Launch the munition with initial conditions."""
        self.position = initial_position
        self.velocity = initial_velocity
        self.orientation = initial_orientation
        self.active = True
        self.target_locked = False
        logger.info(f"Munition {self.id} launched from position {initial_position}")
        
    def set_terrain_model(self, terrain_model: TacticalTerrainModel):
        """Set the terrain model for this munition."""
        self.terrain_model = terrain_model
        logger.info(f"Terrain model set for munition {self.id}")
    
    def update(self, 
              dt: float, 
              target_position: Optional[Position] = None,
              jamming_environment: Optional[Dict[str, Any]] = None):
        """Update munition state for one time step.
        
        Args:
            dt: Time step in seconds
            target_position: Current target position if available
            jamming_environment: Dict containing jamming information
        
        Returns:
            True if munition is still active, False otherwise
        """
        if not self.active:
            return False
            
        # Process electronic warfare effects if applicable
        if jamming_environment and self.vulnerability_model:
            self._process_jamming(jamming_environment)
            
        # Update target tracking with terrain effects
        if target_position:
            self.target_locked = self.seeker.can_track_target(
                target_position, 
                self.position,
                self.orientation,
                jamming_environment.get("jamming_level", 0) if jamming_environment else 0,
                self.terrain_model  # Pass terrain model to seeker
            )
        
        # Calculate guidance commands if target is locked
        if self.target_locked and target_position:
            # Convert to tuples for the guidance system
            current_pos = (self.position.x, self.position.y, self.position.z)
            current_vel = (self.velocity.x, self.velocity.y, self.velocity.z)
            target_pos = (target_position.x, target_position.y, target_position.z)
            
            # Get acceleration commands
            ax, ay, az = self.guidance_system.calculate_guidance_commands(
                current_pos, current_vel, target_pos
            )
        else:
            # No guidance, continue on ballistic trajectory
            ax, ay, az = 0, 0, 0
            
        # Apply aerodynamic forces (simplified)
        speed = self.velocity.magnitude()
        if speed > 0:
            # Drag force
            drag = 0.5 * self.drag_coefficient * 1.225 * speed**2 * self.cross_section
            drag_deceleration = drag / self.mass
            
            # Apply drag in opposite direction of velocity
            vx_norm = self.velocity.x / speed
            vy_norm = self.velocity.y / speed
            vz_norm = self.velocity.z / speed
            
            ax -= vx_norm * drag_deceleration
            ay -= vy_norm * drag_deceleration
            az -= vz_norm * drag_deceleration
            
        # Apply gravity
        az -= 9.81  # m/s²
        
        # Update velocity
        self.velocity.x += ax * dt
        self.velocity.y += ay * dt
        self.velocity.z += az * dt
        
        # Update position
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        self.position.z += self.velocity.z * dt
        
        # Check if munition has hit the ground
        if self.position.z <= 0:
            self.position.z = 0
            self.active = False
            logger.info(f"Munition {self.id} impact at position ({self.position.x}, {self.position.y}, 0)")
            return False
            
        # Update orientation based on velocity (simplified)
        if speed > 1.0:  # Only update if moving significantly
            self.orientation.pitch = -np.arcsin(self.velocity.z / speed)
            self.orientation.yaw = np.arctan2(self.velocity.y, self.velocity.x)
            
        return True
        
    def _process_jamming(self, jamming_environment: Dict[str, Any]):
        """Process electronic warfare effects on the munition."""
        if not self.vulnerability_model:
            return
            
        # Extract jamming parameters
        jamming_power = jamming_environment.get("power", 0)  # dBm
        jamming_frequency = jamming_environment.get("frequency", 0)  # Hz
        jamming_type = jamming_environment.get("type", "noise")
        
        # Check for effects on seeker
        effect = self.vulnerability_model.predict_effect(
            component_id="rf_seeker",
            coupled_power=jamming_power,
            frequency=jamming_frequency,
            modulation_type=jamming_type
        )
        
        # Apply effects
        if effect.effect_type == "upset":
            # Temporary loss of tracking
            self.target_locked = False
            logger.warning(f"Munition {self.id} seeker upset by jamming")
        elif effect.effect_type == "damage":
            # Permanent damage to seeker
            self.target_locked = False
            logger.warning(f"Munition {self.id} seeker damaged by jamming")


def simulate_engagement(
    munition: RFGuidedMunition,
    target_initial_position: Position,
    target_velocity: Velocity,
    simulation_duration: float = 30.0,
    time_step: float = 0.1,
    jamming_environment: Optional[Dict[str, Any]] = None
) -> Tuple[List[Position], List[Position], bool]:
    """
    Simulate an engagement between an RF-guided munition and a target.
    
    Args:
        munition: The RF-guided munition
        target_initial_position: Initial target position
        target_velocity: Target velocity (assumed constant)
        simulation_duration: Total simulation time in seconds
        time_step: Simulation time step in seconds
        jamming_environment: Optional electronic warfare environment
        
    Returns:
        Tuple of (munition_positions, target_positions, hit_success)
    """
    # Initialize result containers
    munition_positions = [Position(
        munition.position.x,
        munition.position.y,
        munition.position.z
    )]
    
    target_position = Position(
        target_initial_position.x,
        target_initial_position.y,
        target_initial_position.z
    )
    
    target_positions = [Position(
        target_position.x,
        target_position.y,
        target_position.z
    )]
    
    hit_success = False
    
    # Run simulation
    time = 0
    while time < simulation_duration and munition.active:
        # Update target position
        target_position.x += target_velocity.x * time_step
        target_position.y += target_velocity.y * time_step
        target_position.z += target_velocity.z * time_step
        
        # Update munition
        munition.update(time_step, target_position, jamming_environment)
        
        # Record positions
        munition_positions.append(Position(
            munition.position.x,
            munition.position.y,
            munition.position.z
        ))
        
        target_positions.append(Position(
            target_position.x,
            target_position.y,
            target_position.z
        ))
        
        # Check for hit
        distance = munition.position.distance_to(target_position)
        if distance < 5.0:  # Hit threshold in meters
            hit_success = True
            logger.info(f"Hit detected at time {time:.1f}s, distance: {distance:.2f}m")
            break
            
        time += time_step
        
    return munition_positions, target_positions, hit_success