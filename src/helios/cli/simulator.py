#!/usr/bin/env python3
"""
Command-line interface for running Helios simulations.
"""

import argparse
import logging
import os
import sys
import yaml
from typing import Dict, Any

from helios.utils.logger import setup_logging
from helios.utils.config_loader import load_config
from helios.simulation.scenario_builder import build_scenario_from_config
from helios.simulation.engine import SimulationEngine
from helios.security.data_sanitizer import DataSanitizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Helios RF Simulation Suite")
    
    parser.add_argument(
        "-c", "--config", 
        required=True,
        help="Path to simulation configuration file"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./results",
        help="Directory to store simulation results"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Enable data sanitization for sensitive information"
    )
    
    parser.add_argument(
        "--sanitize-config",
        help="Path to sanitization configuration file"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the Helios simulator CLI."""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, sanitize=args.sanitize)
    
    # Load sanitization config if specified
    sanitize_config = None
    if args.sanitize and args.sanitize_config:
        try:
            sanitize_config = load_config(args.sanitize_config)
        except Exception as e:
            logging.error(f"Failed to load sanitization config: {e}")
            sys.exit(1)
    
    # Load simulation configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build scenario from config
    try:
        scenario = build_scenario_from_config(args.config)
    except Exception as e:
        logging.error(f"Failed to build scenario: {e}")
        sys.exit(1)
    
    # Initialize simulation engine
    engine = SimulationEngine(scenario=scenario)
    
    # Run simulation
    try:
        logging.info(f"Starting simulation: {scenario.name}")
        # Change from run_simulation to run() which is likely the correct method name
        results = engine.run()
        
        # Save results
        result_path = os.path.join(args.output_dir, f"{scenario.id}_results.json")
        with open(result_path, 'w') as f:
            yaml.dump(results, f)
        
        # Sanitize results if needed
        if args.sanitize:
            from helios.security.result_sanitizer import ResultSanitizer
            sanitizer = ResultSanitizer(sanitize_config)
            sanitized_path = sanitizer.sanitize_file(result_path)
            logging.info(f"Sanitized results saved to: {sanitized_path}")
        
        logging.info(f"Simulation completed. Results saved to: {result_path}")
        
    except Exception as e:
        logging.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()