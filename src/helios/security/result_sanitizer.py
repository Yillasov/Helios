"""Utilities for sanitizing simulation results."""

import os
import json
import csv
import pandas as pd
from typing import Dict, Any, Optional, List, Union

from helios.security.data_sanitizer import DataSanitizer

class ResultSanitizer:
    """
    Utility for sanitizing sensitive information from simulation results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the result sanitizer.
        
        Args:
            config: Optional sanitization configuration
        """
        self.sanitizer = DataSanitizer(config)
    
    def sanitize_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Sanitize a result file based on its extension.
        
        Args:
            input_path: Path to the input file
            output_path: Optional path for the sanitized output file
                         (defaults to input_path with '_sanitized' suffix)
            
        Returns:
            Path to the sanitized file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Default output path if not specified
        if not output_path:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_sanitized{ext}"
        
        # Determine file type and sanitize accordingly
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()
        
        if ext == '.json':
            self._sanitize_json_file(input_path, output_path)
        elif ext == '.csv':
            self._sanitize_csv_file(input_path, output_path)
        elif ext in ['.xlsx', '.xls']:
            self._sanitize_excel_file(input_path, output_path)
        elif ext == '.txt':
            self._sanitize_text_file(input_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return output_path
    
    def _sanitize_json_file(self, input_path: str, output_path: str) -> None:
        """Sanitize a JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Sanitize the data
        if isinstance(data, dict):
            sanitized = self.sanitizer.sanitize_dict(data)
        elif isinstance(data, list):
            sanitized = self.sanitizer.sanitize_list(data)
        else:
            sanitized = data
        
        # Write sanitized data
        with open(output_path, 'w') as f:
            json.dump(sanitized, f, indent=2)
    
    def _sanitize_csv_file(self, input_path: str, output_path: str) -> None:
        """Sanitize a CSV file."""
        # Read CSV into a list of dictionaries
        rows = []
        with open(input_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        # Sanitize each row
        sanitized_rows = [self.sanitizer.sanitize_dict(row) for row in rows]
        
        # Write sanitized data
        if sanitized_rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sanitized_rows[0].keys())
                writer.writeheader()
                writer.writerows(sanitized_rows)
        else:
            # Create empty file with original headers if no rows
            with open(input_path, 'r', newline='') as f_in:
                reader = csv.reader(f_in)
                headers = next(reader, [])
            
            with open(output_path, 'w', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(headers)
    
    def _sanitize_excel_file(self, input_path: str, output_path: str) -> None:
        """Sanitize an Excel file."""
        # Read Excel file
        df = pd.read_excel(input_path)
        
        # Convert to dict, sanitize, and convert back to DataFrame
        data_dict = df.to_dict(orient='records')
        sanitized_dict = self.sanitizer.sanitize_list(data_dict)
        sanitized_df = pd.DataFrame(sanitized_dict)
        
        # Write sanitized data
        sanitized_df.to_excel(output_path, index=False)
    
    def _sanitize_text_file(self, input_path: str, output_path: str) -> None:
        """Sanitize a text file."""
        with open(input_path, 'r') as f:
            text = f.read()
        
        # Sanitize the text
        sanitized = self.sanitizer.sanitize_text(text)
        
        # Write sanitized data
        with open(output_path, 'w') as f:
            f.write(sanitized)