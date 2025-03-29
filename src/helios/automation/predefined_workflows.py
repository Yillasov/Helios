"""Predefined workflows for common Helios processes."""

from typing import Dict, Any, Optional
import os

from helios.automation.workflow_engine import Workflow, WorkflowStep
from helios.core.data_structures import Scenario
from helios.simulation.scenario_builder import build_scenario_from_config
from helios.simulation.engine import SimulationEngine
from helios.formats.standard_formats import SimulationResultSummaryFormat
from helios.utils.logger import get_logger

logger = get_logger(__name__)

def create_simulation_workflow(
    config_path: str,
    output_dir: str,
    duration: float = 10.0,
    record_data: bool = True
) -> Workflow:
    """Create a standard simulation workflow.
    
    Args:
        config_path: Path to scenario configuration
        output_dir: Directory to store outputs
        duration: Simulation duration in seconds
        record_data: Whether to record simulation data
        
    Returns:
        Configured workflow
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the workflow
    workflow = Workflow(
        name="Standard Simulation",
        description=f"Run simulation from {config_path} for {duration}s"
    )
    
    # Add context
    workflow.add_context("config_path", config_path)
    workflow.add_context("output_dir", output_dir)
    workflow.add_context("duration", duration)
    workflow.add_context("record_data", record_data)
    
    # Define steps
    workflow.add_step(
        "load_scenario",
        WorkflowStep(
            name="Load Scenario",
            function=build_scenario_from_config,
            args=[config_path]
        )
    )
    
    workflow.add_step(
        "setup_engine",
        WorkflowStep(
            name="Setup Simulation Engine",
            function=lambda scenario: SimulationEngine(scenario),
            depends_on=["load_scenario"]
        )
    )
    
    workflow.add_step(
        "run_simulation",
        WorkflowStep(
            name="Run Simulation",
            function=lambda engine, duration: engine.run(duration),
            depends_on=["setup_engine"]
        )
    )
    
    workflow.add_step(
        "export_results",
        WorkflowStep(
            name="Export Results",
            function=lambda engine, output_dir: _export_simulation_results(
                engine, output_dir
            ),
            depends_on=["run_simulation"]
        )
    )
    
    return workflow

def _export_simulation_results(engine: SimulationEngine, output_dir: str) -> str:
    """Export simulation results to a file.
    
    Args:
        engine: Simulation engine with results
        output_dir: Directory to store outputs
        
    Returns:
        Path to results file
    """
    # Get the current time as end time
    end_time = engine.current_time
    # Use scenario start time as start time
    start_time = engine.scenario.start_time
    
    results = SimulationResultSummaryFormat(
        run_id=str(engine.scenario.id),
        scenario_name=engine.scenario.name,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=end_time - start_time,
        metrics={
            "num_platforms": len(engine.scenario.platforms),
            # Use safer approach for accessing signals
            "num_signals": len(getattr(engine, '_active_signals', [])),
            # Use safer approach for performance metrics
            "avg_processing_time": getattr(engine, 'avg_processing_time', 0.0)
        }
    )
    
    output_path = os.path.join(output_dir, f"sim_results_{results.run_id}.json")
    
    # Export results
    with open(output_path, 'w') as f:
        import json
        from dataclasses import asdict
        json.dump(asdict(results), f, indent=2)
    
    return output_path

def create_hardware_test_workflow(device_id: str, output_dir: str) -> Workflow:
    """Create a hardware test workflow.
    
    Args:
        device_id: ID of the hardware device to test
        output_dir: Directory to store outputs
        
    Returns:
        Configured workflow
    """
    # Implementation would be similar to simulation workflow
    # but with hardware-specific steps
    workflow = Workflow(
        name="Hardware Test",
        description=f"Test hardware device {device_id}"
    )
    
    # Add steps for hardware testing
    # ...
    
    return workflow