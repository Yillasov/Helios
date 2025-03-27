import pytest
import os
import sys

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Define fixtures here that can be used across multiple test files
@pytest.fixture
def sample_fixture():
    """Example fixture for testing."""
    return True