"""Logging utilities for Helios."""

import logging
import sys
from typing import Optional, Dict, Any

LOGGING_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s (%(filename)s:%(lineno)d)"
)
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: int = logging.INFO, stream=sys.stdout, 
                 sanitize: bool = False, sanitize_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Configures the root logger for the Helios application.

    Args:
        level: The minimum logging level to output (e.g., logging.DEBUG, logging.INFO).
        stream: The output stream (e.g., sys.stdout, sys.stderr, or a file handle).
        sanitize: Whether to sanitize sensitive information in logs.
        sanitize_config: Optional configuration for the sanitizer.
    """
    logging.basicConfig(
        level=level,
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        stream=stream,
        force=True # Override any existing basicConfig by other libraries
    )
    
    # Add sanitization filter if requested
    if sanitize:
        from helios.security.log_sanitizer import SanitizingFilter
        root_logger = logging.getLogger()
        sanitizing_filter = SanitizingFilter(config=sanitize_config)
        root_logger.addFilter(sanitizing_filter)
        logging.info("Log sanitization enabled")
    
    # You could potentially add file handlers here as well
    # handler = logging.FileHandler("helios_run.log")
    # handler.setFormatter(logging.Formatter(LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT))
    # logging.getLogger().addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the specified name.

    Args:
        name: The name for the logger (usually __name__ of the calling module).

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)

# Initialize logging when this module is imported, can be reconfigured later if needed
# setup_logging() # Optional: Configure on import (might have side effects)