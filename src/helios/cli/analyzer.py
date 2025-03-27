#!/usr/bin/env python3
"""
Command-line interface for analyzing Helios simulation results.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

from helios.utils.logger import setup_logging
from helios.utils.config_loader import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Helios RF Analysis Suite")
    
    parser.add_argument(
        "-r", "--results-dir", 
        required=True,
        help="Path to simulation results directory"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./analysis",
        help="Directory to store analysis results"
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
    
    return parser.parse_args()

def main():
    """Main entry point for the Helios analyzer CLI."""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level, sanitize=args.sanitize)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import analyzer module
    try:
        from analysis.simulation_analyzer import run_analysis
    except ImportError:
        logging.error("Failed to import analysis module. Make sure it's installed correctly.")
        sys.exit(1)
    
    # Get list of simulation runs
    try:
        run_ids = [d for d in os.listdir(args.results_dir) 
                  if os.path.isdir(os.path.join(args.results_dir, d))]
        
        if not run_ids:
            logging.warning(f"No simulation runs found in {args.results_dir}")
            sys.exit(0)
        
        # Run analysis for each simulation
        for run_id in run_ids:
            logging.info(f"Analyzing simulation run: {run_id}")
            try:
                run_analysis(run_id)
                logging.info(f"Analysis completed for run: {run_id}")
            except Exception as e:
                logging.error(f"Analysis failed for run {run_id}: {e}", exc_info=True)
        
        logging.info(f"All analyses completed. Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()