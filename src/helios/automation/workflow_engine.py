"""Workflow automation engine for end-to-end processes in Helios."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
import time
import json
import os
from enum import Enum, auto
import logging

from helios.formats.standard_formats import SimulationResultSummaryFormat
from helios.utils.logger import get_logger

logger = get_logger(__name__)

class WorkflowStepStatus(Enum):
    """Status of a workflow step."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    function: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    result: Any = None
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 0
    
    def run(self, context: Dict[str, Any]) -> Any:
        """Run this workflow step with the given context."""
        self.status = WorkflowStepStatus.RUNNING
        try:
            # Update kwargs with context variables if they match parameter names
            updated_kwargs = {**self.kwargs}
            for k, v in context.items():
                if k in self.function.__code__.co_varnames:
                    updated_kwargs[k] = v
            
            # Run the function
            self.result = self.function(*self.args, **updated_kwargs)
            self.status = WorkflowStepStatus.COMPLETED
            return self.result
        except Exception as e:
            logger.error(f"Step '{self.name}' failed: {e}")
            self.status = WorkflowStepStatus.FAILED
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                logger.info(f"Retrying step '{self.name}' (attempt {self.retry_count}/{self.max_retries})")
                return self.run(context)
            return None

class Workflow:
    """Manages a sequence of workflow steps."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a new workflow."""
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.context: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def add_step(self, step_id: str, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps[step_id] = step
    
    def add_context(self, key: str, value: Any) -> None:
        """Add a value to the workflow context."""
        self.context[key] = value
    
    def can_run_step(self, step_id: str) -> bool:
        """Check if a step can be run based on its dependencies."""
        step = self.steps[step_id]
        for dep_id in step.depends_on:
            if dep_id not in self.steps:
                logger.error(f"Step '{step_id}' depends on unknown step '{dep_id}'")
                return False
            if self.steps[dep_id].status != WorkflowStepStatus.COMPLETED:
                return False
        return True
    
    def run(self) -> Dict[str, Any]:
        """Run the entire workflow."""
        self.start_time = time.time()
        logger.info(f"Starting workflow: {self.name}")
        
        # Keep track of steps that have been processed
        processed_steps = set()
        
        # Continue until all steps are processed
        while len(processed_steps) < len(self.steps):
            progress_made = False
            
            for step_id, step in self.steps.items():
                if step_id in processed_steps:
                    continue
                
                if step.status == WorkflowStepStatus.PENDING and self.can_run_step(step_id):
                    logger.info(f"Running step: {step.name}")
                    result = step.run(self.context)
                    self.context[step_id] = result
                    processed_steps.add(step_id)
                    progress_made = True
                elif step.status in [WorkflowStepStatus.COMPLETED, WorkflowStepStatus.FAILED, WorkflowStepStatus.SKIPPED]:
                    processed_steps.add(step_id)
                    progress_made = True
            
            if not progress_made:
                # We're stuck - there must be a dependency cycle or all remaining steps have failed dependencies
                for step_id, step in self.steps.items():
                    if step_id not in processed_steps:
                        logger.error(f"Step '{step_id}' could not be processed, marking as SKIPPED")
                        step.status = WorkflowStepStatus.SKIPPED
                        processed_steps.add(step_id)
        
        self.end_time = time.time()
        logger.info(f"Workflow completed in {self.end_time - self.start_time:.2f} seconds")
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of the workflow."""
        results = {
            "workflow_name": self.name,
            "description": self.description,
            "duration": (self.end_time - self.start_time) if (self.end_time and self.start_time) else None,
            "steps": {}
        }
        
        for step_id, step in self.steps.items():
            results["steps"][step_id] = {
                "name": step.name,
                "status": step.status.name,
                "result_summary": str(step.result)[:100] + "..." if isinstance(step.result, str) and len(str(step.result)) > 100 else step.result
            }
        
        return results
    
    def export_results(self, output_path: str) -> None:
        """Export workflow results to a file."""
        results = self.get_results()
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Exported workflow results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export workflow results: {e}")