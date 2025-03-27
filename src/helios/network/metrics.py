"""Network metrics calculation based on RF conditions."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from helios.network.data_structures import NetworkLink, NetworkPacket, NetworkNode
from helios.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_packet_delivery_ratio(snr_db: float, interference_level_dbm: float = -120.0) -> float:
    """
    Calculate packet delivery ratio based on SNR and interference.
    
    Args:
        snr_db: Signal-to-Noise Ratio in dB
        interference_level_dbm: Interference power level in dBm
        
    Returns:
        Packet delivery ratio (0.0-1.0)
    """
    # Simple sigmoid model for PDR based on SNR
    # PDR approaches 1.0 as SNR increases, and 0.0 as SNR decreases
    base_pdr = 1.0 / (1.0 + np.exp(-0.5 * (snr_db - 10.0)))
    
    # Apply interference penalty (higher interference reduces PDR)
    # Normalize interference to a 0-1 scale (assuming -120 dBm is negligible, -60 dBm is severe)
    interference_factor = np.clip((interference_level_dbm + 120) / 60.0, 0.0, 1.0)
    interference_penalty = 0.3 * interference_factor  # Up to 30% reduction due to interference
    
    # Final PDR with interference effects
    pdr = base_pdr * (1.0 - interference_penalty)
    return np.clip(pdr, 0.0, 1.0)  # Ensure PDR is between 0 and 1

def estimate_link_latency(
    link: NetworkLink, 
    snr_db: float, 
    packet_size_bytes: int = 1500,
    retransmission_enabled: bool = True
) -> float:
    """
    Estimate network latency based on link properties and RF conditions.
    
    Args:
        link: Network link object
        snr_db: Signal-to-Noise Ratio in dB
        packet_size_bytes: Size of packet in bytes
        retransmission_enabled: Whether retransmissions are enabled
        
    Returns:
        Estimated latency in seconds
    """
    # Base latency from link properties
    base_latency = link.latency
    
    # Transmission time based on bandwidth and packet size
    transmission_time = (packet_size_bytes * 8) / link.bandwidth if link.bandwidth > 0 else 0
    
    # Calculate packet loss probability based on SNR
    # Higher SNR means lower packet loss
    packet_loss_prob = 1.0 - calculate_packet_delivery_ratio(snr_db)
    
    # Add latency due to retransmissions if enabled
    retransmission_latency = 0.0
    if retransmission_enabled and packet_loss_prob > 0:
        # Expected number of transmissions = 1 / (1 - p) where p is loss probability
        # We subtract 1 to get the number of retransmissions
        avg_retransmissions = min(1.0 / (1.0 - packet_loss_prob) - 1.0, 5.0)  # Cap at 5 retransmissions
        # Each retransmission adds transmission time plus RTT
        retransmission_latency = avg_retransmissions * (transmission_time + 2 * base_latency)
    
    # Calculate queuing delay based on link utilization
    queuing_delay = 0.0
    if link.current_utilization > 0:
        # Simple M/M/1 queue model: delay increases as utilization approaches 1
        utilization_factor = min(link.current_utilization, 0.95)  # Cap at 0.95 to avoid infinity
        queuing_delay = transmission_time * utilization_factor / (1.0 - utilization_factor)
    
    # Total latency
    total_latency = base_latency + transmission_time + retransmission_latency + queuing_delay
    
    return total_latency

def calculate_network_throughput(
    link: NetworkLink, 
    snr_db: float,
    interference_level_dbm: float = -120.0,
    protocol_overhead: float = 0.1  # 10% overhead by default
) -> float:
    """
    Calculate effective network throughput based on RF conditions.
    
    Args:
        link: Network link object
        snr_db: Signal-to-Noise Ratio in dB
        interference_level_dbm: Interference power level in dBm
        protocol_overhead: Fraction of bandwidth used for protocol overhead
        
    Returns:
        Effective throughput in bits per second
    """
    # Calculate PDR
    pdr = calculate_packet_delivery_ratio(snr_db, interference_level_dbm)
    
    # Calculate effective bandwidth considering link utilization
    available_bandwidth = link.bandwidth * (1.0 - link.current_utilization)
    
    # Apply protocol overhead
    usable_bandwidth = available_bandwidth * (1.0 - protocol_overhead)
    
    # Apply packet delivery ratio to get effective throughput
    effective_throughput = usable_bandwidth * pdr
    
    return effective_throughput

def estimate_rf_impact_on_network(
    link: NetworkLink,
    snr_db: float,
    interference_level_dbm: float = -120.0
) -> Dict[str, float]:
    """
    Comprehensive assessment of RF impact on network performance.
    
    Args:
        link: Network link object
        snr_db: Signal-to-Noise Ratio in dB
        interference_level_dbm: Interference power level in dBm
        
    Returns:
        Dictionary with network performance metrics
    """
    # Standard packet size
    packet_size = 1500  # bytes
    
    # Calculate metrics
    pdr = calculate_packet_delivery_ratio(snr_db, interference_level_dbm)
    latency = estimate_link_latency(link, snr_db, packet_size)
    throughput = calculate_network_throughput(link, snr_db, interference_level_dbm)
    
    # Calculate jitter (simplified model based on SNR variability)
    # Lower SNR tends to cause higher jitter
    jitter_factor = max(0, (20 - snr_db)) / 20 if snr_db < 20 else 0
    jitter = latency * 0.2 * jitter_factor  # Up to 20% of latency
    
    # Estimate reliability (probability of successful transmission within timeout)
    timeout_factor = 3.0  # Timeout is 3x expected latency
    reliability = pdr * (1.0 - np.exp(-timeout_factor))
    
    return {
        "packet_delivery_ratio": pdr,
        "latency": latency,
        "throughput": throughput,
        "jitter": jitter,
        "reliability": reliability,
        "link_quality": min(1.0, snr_db / 30.0) if snr_db > 0 else 0.0  # Normalized link quality
    }