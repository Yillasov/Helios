"""Data sanitization utilities for removing sensitive information from logs and results."""

import re
import logging
import json
import copy
from typing import Dict, List, Any, Union, Optional, Pattern, Callable

logger = logging.getLogger(__name__)

class DataSanitizer:
    """
    Utility for sanitizing sensitive data from logs, results, and other outputs.
    Supports pattern-based redaction and custom sanitization rules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data sanitizer with configuration.
        
        Args:
            config: Optional configuration dictionary with sanitization rules
        """
        # Default patterns to sanitize
        self.patterns: Dict[str, Pattern] = {
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'mac_address': re.compile(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'password': re.compile(r'password["\s:=]+[^"\s,)]+'),
            'api_key': re.compile(r'api[_-]?key["\s:=]+[^"\s,)]+'),
            'token': re.compile(r'token["\s:=]+[^"\s,)]+'),
            'coordinates': re.compile(r'\b(-?\d+\.\d+),\s*(-?\d+\.\d+)\b'),  # lat,long format
        }
        
        # Default replacements
        self.replacements: Dict[str, str] = {
            'ip_address': '[REDACTED_IP]',
            'mac_address': '[REDACTED_MAC]',
            'email': '[REDACTED_EMAIL]',
            'password': 'password="[REDACTED]"',
            'api_key': 'api_key="[REDACTED]"',
            'token': 'token="[REDACTED]"',
            'coordinates': '[REDACTED_COORDINATES]',
        }
        
        # Custom field sanitizers for structured data
        self.field_sanitizers: Dict[str, Callable] = {
            'password': lambda x: '[REDACTED]',
            'secret': lambda x: '[REDACTED]',
            'token': lambda x: '[REDACTED]',
            'api_key': lambda x: '[REDACTED]',
            'private_key': lambda x: '[REDACTED]',
            'location': self._sanitize_location,
            'position': self._sanitize_position,
            'coordinates': self._sanitize_coordinates,
        }
        
        # Load custom configuration if provided
        if config:
            self._load_config(config)
    
    def _load_config(self, config: Dict[str, Any]) -> None:
        """
        Load custom sanitization configuration.
        
        Args:
            config: Configuration dictionary with sanitization rules
        """
        # Add or update patterns
        if 'patterns' in config:
            for name, pattern in config['patterns'].items():
                self.patterns[name] = re.compile(pattern)
        
        # Add or update replacements
        if 'replacements' in config:
            self.replacements.update(config['replacements'])
        
        # Add sensitive field names
        if 'sensitive_fields' in config:
            for field in config['sensitive_fields']:
                if field not in self.field_sanitizers:
                    self.field_sanitizers[field] = lambda x: '[REDACTED]'
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize sensitive information in text data.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return text
            
        sanitized = text
        
        # Apply all patterns
        for pattern_name, pattern in self.patterns.items():
            replacement = self.replacements.get(pattern_name, '[REDACTED]')
            sanitized = pattern.sub(replacement, sanitized)
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize sensitive information in dictionary data.
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        if not data:
            return data
            
        # Create a deep copy to avoid modifying the original
        sanitized = copy.deepcopy(data)
        
        # Process all keys recursively
        for key, value in list(sanitized.items()):
            # Check if this key needs sanitization
            if key.lower() in self.field_sanitizers:
                sanitized[key] = self.field_sanitizers[key.lower()](value)
            # Recursively sanitize nested dictionaries
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            # Recursively sanitize lists
            elif isinstance(value, list):
                sanitized[key] = self.sanitize_list(value)
            # Sanitize string values
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
        
        return sanitized
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Sanitize sensitive information in list data.
        
        Args:
            data: List to sanitize
            
        Returns:
            Sanitized list
        """
        if not data:
            return data
            
        # Create a deep copy to avoid modifying the original
        sanitized = copy.deepcopy(data)
        
        # Process all items recursively
        for i, item in enumerate(sanitized):
            if isinstance(item, dict):
                sanitized[i] = self.sanitize_dict(item)
            elif isinstance(item, list):
                sanitized[i] = self.sanitize_list(item)
            elif isinstance(item, str):
                sanitized[i] = self.sanitize_text(item)
        
        return sanitized
    
    def sanitize_json(self, json_data: str) -> str:
        """
        Sanitize sensitive information in JSON data.
        
        Args:
            json_data: JSON string to sanitize
            
        Returns:
            Sanitized JSON string
        """
        try:
            # Parse JSON
            data = json.loads(json_data)
            
            # Sanitize the data structure
            if isinstance(data, dict):
                sanitized = self.sanitize_dict(data)
            elif isinstance(data, list):
                sanitized = self.sanitize_list(data)
            else:
                # Simple value, just return as is
                return json_data
            
            # Convert back to JSON
            return json.dumps(sanitized, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, treat as text
            logger.warning("Invalid JSON provided for sanitization, treating as text")
            return self.sanitize_text(json_data)
    
    def sanitize_log_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """
        Sanitize sensitive information in a log record.
        
        Args:
            record: Log record to sanitize
            
        Returns:
            Sanitized log record
        """
        # Create a copy of the record to avoid modifying the original
        sanitized_record = copy.copy(record)
        
        # Sanitize the message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            sanitized_record.msg = self.sanitize_text(record.msg)
        
        # Sanitize the args if they exist
        if hasattr(record, 'args') and record.args:
            args_list = list(record.args)
            for i, arg in enumerate(args_list):
                if isinstance(arg, str):
                    args_list[i] = self.sanitize_text(arg)
                elif isinstance(arg, dict):
                    args_list[i] = self.sanitize_dict(arg)
                elif isinstance(arg, list):
                    args_list[i] = self.sanitize_list(arg)
            sanitized_record.args = tuple(args_list)
        
        return sanitized_record
    
    def _sanitize_location(self, location: Any) -> Any:
        """Sanitize location data."""
        if isinstance(location, dict):
            sanitized = copy.deepcopy(location)
            if 'latitude' in sanitized:
                sanitized['latitude'] = 0.0
            if 'longitude' in sanitized:
                sanitized['longitude'] = 0.0
            return sanitized
        return '[REDACTED_LOCATION]'
    
    def _sanitize_position(self, position: Any) -> Any:
        """Sanitize position data."""
        if isinstance(position, dict):
            sanitized = copy.deepcopy(position)
            if 'lat' in sanitized:
                sanitized['lat'] = 0.0
            if 'lon' in sanitized:
                sanitized['lon'] = 0.0
            if 'latitude' in sanitized:
                sanitized['latitude'] = 0.0
            if 'longitude' in sanitized:
                sanitized['longitude'] = 0.0
            return sanitized
        return '[REDACTED_POSITION]'
    
    def _sanitize_coordinates(self, coords: Any) -> Any:
        """Sanitize coordinate data."""
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return [0.0, 0.0] + list(coords[2:])
        return '[REDACTED_COORDINATES]'