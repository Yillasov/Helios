"""RF Propagation Models."""

import numpy as np
import math
from scipy.constants import c as SPEED_OF_LIGHT # Correct import `c` instead of `speed_of_light`

from helios.core.interfaces import IPropagationModel
from helios.core.data_structures import Position, Velocity, Signal, Platform, EnvironmentParameters # Added Velocity, Platform
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class LOSPropagationModel(IPropagationModel):
    """
    A simple Line-of-Sight (LOS) propagation model using Free Space Path Loss (FSPL)
    and calculating Doppler shift based on relative velocities.
    Assumes unobstructed path between transmitter and receiver.
    """

    def calculate_path_loss(self,
                           tx_position: Position,
                           rx_position: Position,
                           frequency: float,
                           environment: EnvironmentParameters) -> float:
        """
        Calculate path loss using the Free Space Path Loss (FSPL) formula.

        Args:
            tx_position: Transmitter position.
            rx_position: Receiver position.
            frequency: Signal frequency in Hz.
            environment: Environmental parameters (not used in basic FSPL).

        Returns:
            Path loss in dB. Returns positive infinity if distance or frequency is non-positive.
        """
        distance = tx_position.distance_to(rx_position)

        if distance <= 1e-6 or frequency <= 0:
            logger.warning(f"Invalid input for FSPL: distance={distance:.2f}m, freq={frequency:.2f}Hz. Returning inf loss.")
            return float('inf') # Return infinity for invalid inputs or zero distance

        # FSPL formula in dB: 20 * log10(d) + 20 * log10(f) - 147.55
        # Formula updated using speed of light for better precision constant
        # FSPL (dB) = 20*log10(distance) + 20*log10(frequency) + 20*log10(4*pi/c)
        fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * math.pi / SPEED_OF_LIGHT)
        # fspl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55 # Old constant approximation

        logger.debug(f"FSPL calculation: dist={distance:.2f}m, freq={frequency:.2f}Hz -> Loss={fspl_db:.2f}dB")

        return max(0.0, fspl_db) # Path loss cannot be negative

    def apply_propagation_effects(self,
                                 signal: Signal,
                                 rx_platform: Platform, # Updated parameter
                                 environment: EnvironmentParameters) -> Signal:
        """
        Apply LOS propagation effects (path loss, delay, Doppler) to a signal.

        Modifies the signal's power based on FSPL, calculates propagation delay,
        and calculates Doppler shift based on relative radial velocity.

        Args:
            signal: Original transmitted signal, containing origin pos, source velocity, freq, power.
            rx_platform: The receiving platform, providing its current position and velocity.
            environment: Environmental parameters.

        Returns:
            A *new* Signal object with updated power (representing received power),
            calculated propagation delay, and calculated doppler_shift.
        """
        tx_position = signal.origin
        tx_velocity = signal.source_velocity # Get source velocity from signal
        rx_position = rx_platform.position # Get receiver position from platform
        rx_velocity = rx_platform.velocity # Get receiver velocity from platform
        frequency = signal.waveform.center_frequency

        # --- Calculate Distance and Path Loss ---
        distance = tx_position.distance_to(rx_position)
        path_loss_db = self.calculate_path_loss(tx_position, rx_position, frequency, environment)
        received_power_dbm = signal.power - path_loss_db

        # --- Calculate Propagation Delay ---
        delay = distance / SPEED_OF_LIGHT if SPEED_OF_LIGHT > 0 and distance > 1e-6 else 0.0

        # --- Calculate Doppler Shift ---
        doppler_shift = 0.0
        if distance > 1e-6 and SPEED_OF_LIGHT > 0: # Avoid division by zero
            # Calculate relative velocity vector (Rx - Tx)
            rel_vel_vec = np.array([rx_velocity.x - tx_velocity.x,
                                    rx_velocity.y - tx_velocity.y,
                                    rx_velocity.z - tx_velocity.z])

            # Calculate line-of-sight (LOS) unit vector (from Tx to Rx)
            los_vec = np.array([rx_position.x - tx_position.x,
                                rx_position.y - tx_position.y,
                                rx_position.z - tx_position.z])
            los_unit_vec = los_vec / distance

            # Calculate relative speed along LOS (projection of relative velocity onto LOS vector)
            # Note: A positive value means the objects are moving apart.
            #       A negative value means the objects are moving closer.
            relative_speed_los = np.dot(rel_vel_vec, los_unit_vec)

            # Calculate Doppler shift: fd = - (v_relative / c) * f
            # The negative sign ensures that closing speed (negative relative_speed_los) gives positive shift.
            doppler_shift = - (relative_speed_los / SPEED_OF_LIGHT) * frequency
        else:
             logger.warning(f"Cannot calculate Doppler shift for signal {signal.id}: distance or speed of light is zero.")


        logger.debug(
            f"Applied propagation to signal {signal.id}: "
            f"RxPower={received_power_dbm:.2f}dBm, Delay={delay*1e6:.2f}us, "
            f"Doppler={doppler_shift:.2f}Hz"
        )

        # Create a modified copy of the signal
        import dataclasses
        modified_signal = dataclasses.replace(
            signal,
            power=received_power_dbm,
            propagation_delay=delay,
            doppler_shift=doppler_shift # Store the calculated doppler
        )

        return modified_signal