"""Integration of multistatic RCS models with the existing RCS framework."""

from typing import Dict, Optional, Tuple, List, Any, Union
import os
import numpy as np

from helios.core.data_structures import Position, Orientation
from helios.environment.rcs import RCSModel, RCSModelType
from helios.environment.rcs_multistatic import MultistaticRCSModel, RCSDataFormat
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class ImportedRCSModel(RCSModel):
    """
    RCS model that uses imported data from external tools/measurements.
    Integrates with the existing RCS framework.
    """
    
    def __init__(self, 
                data_file: str, 
                format_type: RCSDataFormat = RCSDataFormat.CSV,
                name: str = "imported_rcs"):
        """
        Initialize the imported RCS model.
        
        Args:
            data_file: Path to the RCS data file
            format_type: Format of the data file
            name: Name identifier for this RCS model
        """
        super().__init__(RCSModelType.COMPLEX)
        self.name = name
        self.multistatic_model = MultistaticRCSModel(name)
        
        # Import data
        success = self.multistatic_model.import_data(data_file, format_type)
        if not success:
            logger.warning(f"Failed to import RCS data from {data_file}. Using default values.")
    
    def calculate_rcs(self, 
                     frequency: float, 
                     target_orientation: Orientation,
                     incident_direction: Tuple[float, float, float]) -> float:
        """
        Calculate RCS for given parameters.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            
        Returns:
            RCS value in m²
        """
        # Use the multistatic model for monostatic RCS calculation
        return self.multistatic_model.calculate_rcs(
            frequency, 
            target_orientation, 
            incident_direction
        )
    
    def calculate_bistatic_rcs(self,
                              frequency: float,
                              target_orientation: Orientation,
                              incident_direction: Tuple[float, float, float],
                              observation_direction: Tuple[float, float, float]) -> float:
        """
        Calculate bistatic RCS.
        
        Args:
            frequency: Signal frequency in Hz
            target_orientation: Orientation of the target
            incident_direction: Direction of incident wave (unit vector)
            observation_direction: Direction to observer (unit vector)
            
        Returns:
            RCS value in m²
        """
        return self.multistatic_model.calculate_rcs(
            frequency,
            target_orientation,
            incident_direction,
            observation_direction
        )


def create_rcs_model_from_file(file_path: str, 
                              format_type: Optional[RCSDataFormat] = None,
                              name: Optional[str] = None) -> RCSModel:
    """
    Factory function to create an RCS model from an external data file.
    
    Args:
        file_path: Path to the RCS data file
        format_type: Format of the data file (auto-detected if None)
        name: Name for the model (derived from filename if None)
        
    Returns:
        An RCS model instance
    """
    # Auto-detect format if not specified
    if format_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            format_type = RCSDataFormat.CSV
        elif ext == '.json':
            format_type = RCSDataFormat.JSON
        elif ext == '.npy':
            format_type = RCSDataFormat.NPY
        else:
            format_type = RCSDataFormat.CSV
            logger.warning(f"Unknown file extension: {ext}. Assuming CSV format.")
    
    # Auto-generate name if not specified
    if name is None:
        name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create and return the model
    return ImportedRCSModel(file_path, format_type, name)