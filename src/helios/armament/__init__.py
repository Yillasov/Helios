"""Armament systems module for Helios RF suite.

This module provides specialized components and tools for designing
and analyzing RF-based armament systems.
"""

from helios.armament.targeting import TargetingSystem, TargetDesignation
from helios.armament.guidance import GuidanceSystem, TrajectoryCalculator
from helios.armament.effects import EffectsModel, CollateralAnalyzer
from helios.armament.hardening import HardeningProfile, EMPProtection
from helios.armament.design import ArmamentDesign
from helios.armament.emp_analysis import EMPHardeningAnalyzer, EMPThreatProfile, STANDARD_EMP_THREATS
from helios.armament.battlefield_spectrum import (
    BattlefieldSpectrumAnalyzer, BattlefieldSpectrumManager,
    FrequencyBand, BandStatus, SpectrumThreatLevel
)
from helios.armament.jamming_predictor import (
    JammingPredictor, JammingScenario, JammingEffectiveness, JammerType
)
from helios.armament.acquisition_calculator import (
    TargetAcquisitionCalculator, AcquisitionParameters, AcquisitionMode
)
from helios.armament.eob_analysis import (
    ElectronicOrderOfBattle, EOBAnalyzer, EmitterRecord,
    EmitterConfidence, EmitterStatus, CollectionResult, 
    SIGINTTarget, SIGINTCollectionModel, CollectionPriority
)

__all__ = [
    'TargetingSystem', 'TargetDesignation',
    'GuidanceSystem', 'TrajectoryCalculator',
    'EffectsModel', 'CollateralAnalyzer',
    'HardeningProfile', 'EMPProtection',
    'ArmamentDesign',
    'EMPHardeningAnalyzer', 'EMPThreatProfile', 'STANDARD_EMP_THREATS',
    'BattlefieldSpectrumAnalyzer', 'BattlefieldSpectrumManager',
    'FrequencyBand', 'BandStatus', 'SpectrumThreatLevel',
    'JammingPredictor', 'JammingScenario', 'JammingEffectiveness', 'JammerType',
    'TargetAcquisitionCalculator', 'AcquisitionParameters', 'AcquisitionMode',
    'ElectronicOrderOfBattle', 'EOBAnalyzer', 'EmitterRecord',
    'EmitterConfidence', 'EmitterStatus', 'CollectionResult',
    'SIGINTTarget', 'SIGINTCollectionModel', 'CollectionPriority'
]