"""Database for correlating waveforms, targets, and electromagnetic effects."""

import sqlite3
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import pandas as pd
from datetime import datetime

from helios.core.data_structures import HPMWaveform, Signal
from helios.environment.hpm_coupling import HPMEffect
from helios.effects.susceptibility import ComponentSusceptibility, EffectType
from helios.waveforms.agile_hpm import AgileHPMParameters
from helios.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EffectRecord:
    """Record of an HPM effect for database storage."""
    id: str  # Unique identifier
    timestamp: float  # When the effect occurred
    waveform_id: str  # ID of the waveform that caused the effect
    target_id: str  # ID of the affected target/component
    effect_type: str  # Type of effect (upset, burnout, etc.)
    severity: float  # Severity level (0-1)
    duration: float  # Effect duration in seconds
    
    # Waveform parameters
    frequency: float  # Center frequency in Hz
    peak_power: float  # Peak power in Watts
    pulse_width: Optional[float] = None  # Pulse width in seconds
    modulation_type: Optional[str] = None  # Modulation type if applicable
    
    # Agile parameters (if applicable)
    frequency_pattern: Optional[str] = None  # fixed, hop, sweep, chirp
    amplitude_pattern: Optional[str] = None  # fixed, modulated, staggered
    pw_pattern: Optional[str] = None  # fixed, staggered, modulated
    
    # Target parameters
    target_type: str = ""  # Type of target component
    distance: Optional[float] = None  # Distance in meters
    orientation: Optional[str] = None  # Target orientation
    
    # Additional metadata
    measured: bool = False  # Whether this was measured (True) or predicted (False)
    notes: str = ""  # Additional notes
    
    # Raw data references
    raw_data_path: Optional[str] = None  # Path to raw measurement data if available


class EffectsDatabase:
    """Database for storing and querying HPM effects data."""
    
    # Update the type hint for db_path to Optional[str]
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the effects database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        if db_path is None:
            # Default to data directory in project
            db_path = os.path.join(os.path.expanduser("~"), "Helios", "data", "effects_database.sqlite")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._initialize_database()
        logger.info(f"Initialized effects database at {db_path}")
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create effects table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS effects (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            waveform_id TEXT,
            target_id TEXT,
            effect_type TEXT,
            severity REAL,
            duration REAL,
            frequency REAL,
            peak_power REAL,
            pulse_width REAL,
            modulation_type TEXT,
            frequency_pattern TEXT,
            amplitude_pattern TEXT,
            pw_pattern TEXT,
            target_type TEXT,
            distance REAL,
            orientation TEXT,
            measured BOOLEAN,
            notes TEXT,
            raw_data_path TEXT,
            metadata TEXT
        )
        ''')
        
        # Create index for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_waveform ON effects (waveform_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_target ON effects (target_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_effect_type ON effects (effect_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frequency ON effects (frequency)')
        
        conn.commit()
        conn.close()
    
    def add_effect(self, effect: EffectRecord):
        """Add an effect record to the database.
        
        Args:
            effect: The effect record to add
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert dataclass to dict and handle None values
        effect_dict = asdict(effect)
        
        # Store any additional fields as JSON in metadata
        metadata = {}
        for key, value in effect_dict.items():
            if key not in [field.name for field in EffectRecord.__dataclass_fields__.values()]:
                metadata[key] = value
        
        # Convert metadata to JSON
        effect_dict['metadata'] = json.dumps(metadata)
        
        # Insert into database
        placeholders = ', '.join(['?'] * len(effect_dict))
        columns = ', '.join(effect_dict.keys())
        values = list(effect_dict.values())
        
        cursor.execute(f'''
        INSERT OR REPLACE INTO effects ({columns})
        VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()
        logger.debug(f"Added effect record {effect.id} to database")
    
    def add_effect_from_simulation(
        self,
        effect: HPMEffect,
        waveform: HPMWaveform,
        target: ComponentSusceptibility,
        distance: Optional[float] = None,
        measured: bool = False
    ):
        """Add an effect from simulation results.
        
        Args:
            effect: The HPM effect
            waveform: The waveform that caused the effect
            target: The target component
            distance: Distance between source and target (meters)
            measured: Whether this was measured or predicted
        """
        # Extract agile parameters if available
        agile_params = None
        if hasattr(waveform, 'pulse_shape_params') and 'agile_params' in waveform.pulse_shape_params:
            agile_params = waveform.pulse_shape_params['agile_params']
        
        # Create effect record
        record = EffectRecord(
            id=f"{waveform.id}_{target.component_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now().timestamp(),
            waveform_id=waveform.id,
            target_id=target.component_id,
            effect_type=effect.effect_type,
            severity=effect.severity,
            duration=effect.duration,
            frequency=waveform.center_frequency,
            peak_power=waveform.peak_power,
            pulse_width=getattr(waveform, 'pulse_width', None),
            modulation_type=str(getattr(waveform, 'modulation_type', None)),
            frequency_pattern=agile_params.get('frequency_pattern', None) if agile_params else None,
            amplitude_pattern=agile_params.get('amplitude_pattern', None) if agile_params else None,
            pw_pattern=agile_params.get('pw_pattern', None) if agile_params else None,
            target_type=target.component_type,
            distance=distance,
            measured=measured
        )
        
        self.add_effect(record)
    
    def query_effects(self, **kwargs) -> pd.DataFrame:
        """Query effects with flexible filtering.
        
        Args:
            **kwargs: Filter criteria (e.g., waveform_id='wave1', effect_type='upset')
            
        Returns:
            DataFrame of matching effects
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = "SELECT * FROM effects"
        params = []
        
        if kwargs:
            conditions = []
            for key, value in kwargs.items():
                if isinstance(value, (list, tuple)):
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute query and return as DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def find_effective_waveforms(self, target_type: Optional[str] = None, effect_type: Optional[str] = None) -> pd.DataFrame:
        """Find most effective waveforms for a given target type.
        
        Args:
            target_type: Type of target component
            effect_type: Type of effect to consider
            
        Returns:
            DataFrame of waveforms ranked by effectiveness
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = """
        SELECT 
            waveform_id,
            frequency,
            peak_power,
            pulse_width,
            modulation_type,
            frequency_pattern,
            amplitude_pattern,
            pw_pattern,
            COUNT(*) as effect_count,
            AVG(severity) as avg_severity
        FROM effects
        """
        
        params = []
        conditions = []
        
        if target_type:
            conditions.append("target_type = ?")
            params.append(target_type)
        
        if effect_type:
            conditions.append("effect_type = ?")
            params.append(effect_type)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
        GROUP BY waveform_id, frequency, peak_power, pulse_width, 
                 modulation_type, frequency_pattern, amplitude_pattern, pw_pattern
        ORDER BY avg_severity DESC, effect_count DESC
        """
        
        # Execute query and return as DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    # Update the type hint for target_type to Optional[str]
    def analyze_frequency_effectiveness(self, target_type: Optional[str] = None) -> pd.DataFrame:
        """Analyze effectiveness of different frequencies.
        
        Args:
            target_type: Type of target component
            
        Returns:
            DataFrame with frequency bands and effectiveness metrics
        """
        df = self.query_effects(target_type=target_type) if target_type else self.query_effects()
        
        if df.empty:
            return pd.DataFrame()
        
        # Create frequency bands
        # Ensure 'frequency' is numeric, coercing errors to NaN
        df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
        # Drop rows where frequency could not be converted
        df = df.dropna(subset=['frequency'])

        if df.empty:
             logger.warning("No valid numeric frequency data found for frequency effectiveness analysis.")
             return pd.DataFrame()

        # Define bins dynamically or use fixed bins appropriate for the expected range
        try:
             # Attempt using quantiles for dynamic binning if data varies widely
             df['frequency_band'] = pd.qcut(df['frequency'], q=10, labels=[f"Band {i+1}" for i in range(10)], duplicates='drop')
        except ValueError:
             # Fallback to fixed bins or simpler logic if qcut fails (e.g., not enough unique values)
             logger.warning("Could not create 10 quantile bins for frequency analysis, using fixed-width bins.")
             df['frequency_band'] = pd.cut(df['frequency'], bins=10, labels=[f"Band {i+1}" for i in range(10)], include_lowest=True)


        # Ensure 'severity' is numeric
        df['severity'] = pd.to_numeric(df['severity'], errors='coerce')

        # Group by frequency band
        # Use observed=True in groupby if using pandas >= 1.1 to avoid issues with unused categories
        analysis = df.groupby('frequency_band', observed=True).agg(
            effect_count=('id', 'count'),
            avg_severity=('severity', 'mean'),
            min_freq=('frequency', 'min'),
            max_freq=('frequency', 'max'),
            avg_freq=('frequency', 'mean'),
            effect_types=('effect_type', lambda x: x.value_counts().to_dict())
        ).reset_index()

        # Rename columns if multi-index was created (depends on pandas version)
        # analysis.columns = ['frequency_band', 'effect_count', 'avg_severity',
        #                    'min_freq', 'max_freq', 'avg_freq', 'effect_types']
        
        return analysis
    
    def analyze_agile_parameters(self) -> Dict[str, pd.DataFrame]:
        """Analyze effectiveness of different agile parameter combinations.
        
        Returns:
            Dictionary of DataFrames with analysis results
        """
        df = self.query_effects()
        
        if df.empty:
            return {}
        
        results = {}
        
        # Analyze frequency patterns
        freq_analysis = df.groupby('frequency_pattern').agg({
            'id': 'count',
            'severity': 'mean'
        }).reset_index()
        freq_analysis.columns = ['frequency_pattern', 'effect_count', 'avg_severity']
        results['frequency_patterns'] = freq_analysis
        
        # Analyze amplitude patterns
        amp_analysis = df.groupby('amplitude_pattern').agg({
            'id': 'count',
            'severity': 'mean'
        }).reset_index()
        amp_analysis.columns = ['amplitude_pattern', 'effect_count', 'avg_severity']
        results['amplitude_patterns'] = amp_analysis
        
        # Analyze pulse width patterns
        pw_analysis = df.groupby('pw_pattern').agg({
            'id': 'count',
            'severity': 'mean'
        }).reset_index()
        pw_analysis.columns = ['pw_pattern', 'effect_count', 'avg_severity']
        results['pw_patterns'] = pw_analysis
        
        # Combined pattern analysis
        combined = df.groupby(['frequency_pattern', 'amplitude_pattern', 'pw_pattern']).agg({
            'id': 'count',
            'severity': 'mean'
        }).reset_index()
        combined.columns = ['frequency_pattern', 'amplitude_pattern', 'pw_pattern', 
                           'effect_count', 'avg_severity']
        results['combined_patterns'] = combined.sort_values('avg_severity', ascending=False)
        
        return results
    
    def export_to_csv(self, output_path: str):
        """Export the entire database to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        df = self.query_effects()
        df.to_csv(output_path, index=False)
        logger.info(f"Exported effects database to {output_path}")