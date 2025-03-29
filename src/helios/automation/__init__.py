"""Workflow automation package for Helios."""

from helios.automation.workflow_engine import (
    Workflow, 
    WorkflowStep,
    WorkflowStepStatus
)
from helios.automation.predefined_workflows import (
    create_simulation_workflow,
    create_hardware_test_workflow
)

__all__ = [
    'Workflow',
    'WorkflowStep',
    'WorkflowStepStatus',
    'create_simulation_workflow',
    'create_hardware_test_workflow'
]