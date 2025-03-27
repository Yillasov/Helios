"""Log sanitization filter for Helios."""

import logging
from typing import Optional, Dict, Any

from helios.security.data_sanitizer import DataSanitizer

class SanitizingFilter(logging.Filter):
    """
    Logging filter that sanitizes sensitive information from log records.
    """
    
    def __init__(self, name: str = '', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sanitizing filter.
        
        Args:
            name: Name of the filter
            config: Optional sanitization configuration
        """
        super().__init__(name)
        self.sanitizer = DataSanitizer(config)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records by sanitizing sensitive information.
        
        Args:
            record: Log record to filter
            
        Returns:
            Always True (allows the record to be logged after sanitization)
        """
        # Sanitize the record
        sanitized_record = self.sanitizer.sanitize_log_record(record)
        
        # Update the original record with sanitized values
        record.msg = sanitized_record.msg
        record.args = sanitized_record.args
        
        return True