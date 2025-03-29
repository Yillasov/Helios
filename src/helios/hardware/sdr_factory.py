"""Factory for creating SDR hardware interfaces."""

from typing import Dict, Any, Optional

from helios.hardware.interfaces import IRadioHardwareInterface
from helios.hardware.usrp_interface import USRPInterface
from helios.hardware.hackrf_interface import HackRFInterface
from helios.hardware.rtlsdr_interface import RTLSDRInterface
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class SDRFactory:
    """Factory for creating SDR hardware interfaces."""
    
    @staticmethod
    def create_sdr(sdr_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[IRadioHardwareInterface]:
        """
        Create an SDR interface of the specified type.
        
        Args:
            sdr_type: Type of SDR ('usrp', 'hackrf', 'rtlsdr')
            config: Optional configuration parameters
            
        Returns:
            SDR interface instance or None if type is invalid
        """
        config = config or {}
        
        if sdr_type.lower() == 'usrp':
            logger.info("Creating USRP interface")
            return USRPInterface()
        elif sdr_type.lower() == 'hackrf':
            logger.info("Creating HackRF interface")
            return HackRFInterface()
        elif sdr_type.lower() in ['rtlsdr', 'rtl-sdr']:
            logger.info("Creating RTL-SDR interface")
            return RTLSDRInterface()
        else:
            logger.error(f"Unknown SDR type: {sdr_type}")
            return None