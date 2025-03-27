
# Helios : Advanced software suite for tactical RF system engineering

Advanced Radio Frequency (RF) systems simulation and analysis suite designed for research, development, testing, and evaluation in complex electromagnetic environments. Helios provides high-fidelity modeling, cognitive capabilities, and flexible integration options.

## Overview

Helios is a Python-based simulation platform enabling sophisticated modeling of RF systems, including transmitters, receivers, antennas, and the propagation environment. It supports the simulation of complex scenarios involving multiple platforms, dynamic signal interactions, interference, and cognitive adaptation strategies. The suite is designed for applications in cognitive radio, electronic warfare, communications systems analysis, and radar systems research.

## Key Features

*   **High-Fidelity RF Environment:** Simulates complex electromagnetic environments, including signal propagation, interference, noise, and clutter (<mcfile name="clutter.py" path="/Users/yessine/Helios/src/helios/environment/clutter.py"></mcfile>).
*   **Cognitive Waveform Generation:** Design and evaluation of adaptive waveforms using a built-in cognitive engine (<mcfile name="engine.py" path="/Users/yessine/Helios/src/helios/cognitive/engine.py"></mcfile>).
*   **Agile High-Power Microwave (HPM) Modeling:** Simulation of HPM waveform generation and basic coupling effects (<mcfile name="hpm_coupling.py" path="/Users/yessine/Helios/src/helios/environment/hpm_coupling.py"></mcfile>).
*   **Discrete-Event Simulation Engine:** Efficiently handles complex scenarios with numerous interacting components and events (<mcsymbol name="ISimulationEngine" filename="interfaces.py" path="/Users/yessine/Helios/src/helios/core/interfaces.py" startline="171" type="class"></mcsymbol>). Includes optimized versions (<mcfile name="optimized_engine.py" path="/Users/yessine/Helios/src/helios/simulation/optimized_engine.py"></mcfile>).
*   **Dynamic Environment Modeling:** Supports time-varying elements like temporal RCS (<mcfile name="temporal_rcs.py" path="/Users/yessine/Helios/src/helios/environment/temporal_rcs.py"></mcfile>) and clutter (<mcfile name="temporal_clutter.py" path="/Users/yessine/Helios/src/helios/environment/temporal_clutter.py"></mcfile>).
*   **Network Control & Analysis:** Tools for simulating and analyzing networked RF systems (<mcfile name="control.py" path="/Users/yessine/Helios/src/helios/network/control.py"></mcfile>).
*   **Hardware Interface Abstraction:** Defines interfaces for potential Hardware-in-the-Loop (HIL) integration (<mcsymbol name="IRadioHardwareInterface" filename="interfaces.py" path="/Users/yessine/Helios/src/helios/hardware/interfaces.py" startline="10" type="class"></mcsymbol>).
*   **Data Sanitization:** Utilities for removing sensitive information from logs and results (<mcfile name="data_sanitizer.py" path="/Users/yessine/Helios/src/helios/security/data_sanitizer.py"></mcfile>).
*   **Extensible Architecture:** Modular design based on interfaces (<mcfile name="interfaces.py" path="/Users/yessine/Helios/src/helios/core/interfaces.py"></mcfile>) allows for easy extension and customization.
*   **Packaging & Deployment:** Ready for deployment using standard Python packaging (<mcfile name="setup.py" path="/Users/yessine/Helios/setup.py"></mcfile>), Docker (<mcfile name="Dockerfile" path="/Users/yessine/Helios/Dockerfile"></mcfile>), and a deployment script (<mcfile name="deploy.sh" path="/Users/yessine/Helios/deploy.sh"></mcfile>).

## Architecture Overview

Helios is built around several core components:

*   **Core:** Defines fundamental data structures (<mcfile name="data_structures.py" path="/Users/yessine/Helios/src/helios/core/data_structures.py"></mcfile>) and interfaces (<mcfile name="interfaces.py" path="/Users/yessine/Helios/src/helios/core/interfaces.py"></mcfile>).
*   **Simulation:** Contains the simulation engine implementations (<mcsymbol name="ISimulationEngine" filename="interfaces.py" path="/Users/yessine/Helios/src/helios/core/interfaces.py" startline="171" type="class"></mcsymbol>, <mcfile name="optimized_engine.py" path="/Users/yessine/Helios/src/helios/simulation/optimized_engine.py"></mcfile>).
*   **Environment:** Models the physical world, including propagation, clutter, RCS (<mcfile name="rcs_multistatic.py" path="/Users/yessine/Helios/src/helios/environment/rcs_multistatic.py"></mcfile>), and HPM effects.
*   **Waveforms:** Manages waveform generation, including cognitive adaptation (<mcfile name="cognitive_generator.py" path="/Users/yessine/Helios/src/helios/waveforms/cognitive_generator.py"></mcfile>).
*   **Hardware:** Provides interfaces and potential implementations for interacting with hardware (<mcfile name="interfaces.py" path="/Users/yessine/Helios/src/helios/hardware/interfaces.py"></mcfile>).
*   **Network:** Simulates network links and control mechanisms (<mcfile name="interfaces.py" path="/Users/yessine/Helios/src/helios/network/interfaces.py"></mcfile>).
*   **Cognitive:** Implements the cognitive engine for intelligent decision-making (<mcfile name="engine.py" path="/Users/yessine/Helios/src/helios/cognitive/engine.py"></mcfile>).
*   **Security:** Includes utilities for authentication, access control, and data sanitization (<mcfile name="auth.py" path="/Users/yessine/Helios/src/helios/security/auth.py"></mcfile>, <mcfile name="data_sanitizer.py" path="/Users/yessine/Helios/src/helios/security/data_sanitizer.py"></mcfile>).
*   **Analysis:** Contains scripts and tools for processing simulation results (<mcfile name="simulation_analyzer.py" path="/Users/yessine/Helios/src/analysis/simulation_analyzer.py"></mcfile>).
*   **CLI:** Provides command-line interfaces for running simulations (<mcfile name="simulator.py" path="/Users/yessine/Helios/src/helios/cli/simulator.py"></mcfile>) and analysis (<mcfile name="analyzer.py" path="/Users/yessine/Helios/src/helios/cli/analyzer.py"></mcfile>).

## Installation

You can install Helios using Poetry (recommended) or standard pip.

**Prerequisites:**
*   Python 3.9+
*   Poetry (optional, for development)

**Using Poetry (for development):**

```bash
# Clone the repository
git clone <repository-url>
cd Helios

# Install dependencies and the project in editable mode
poetry install
```

**Using pip:**

```bash
# Clone the repository
git clone <repository-url>
cd Helios

# Install dependencies and the project (editable mode)
pip install -e .
```

## Configuration

Helios uses YAML and JSON files for configuration, typically located in the `/Users/yessine/Helios/config` directory. Key files include:

*   `default_sim_config.yaml`: Main simulation scenario parameters.
*   `ml_models.json`: Paths and configurations for machine learning models.
*   Sanitization config (optional): Defines rules for data sanitization.

## Usage

Helios provides command-line tools for common tasks.

**Running a Simulation:**

Use the `helios-sim` command, specifying a configuration file.

```bash
# Example: Run simulation using default config, output to 'results/'
helios-sim -c config/default_sim_config.yaml -o results/

# Run with debug logging and data sanitization
helios-sim --log-level DEBUG --sanitize -c config/default_sim_config.yaml -o results_sanitized/
```

**Analyzing Results:**

Use the `helios-analyze` command, pointing to the results directory.

```bash
# Example: Analyze results from the 'results/' directory, output to 'analysis/'
helios-analyze -r results/ -o analysis/
```

**Python API:**

Helios components can also be used directly within Python scripts for more complex workflows or integration.

```python
from helios.simulation.engine import SimulationEngine # Or OptimizedSimulationEngine
from helios.simulation.scenario_builder import build_scenario_from_config

# Load scenario and initialize engine
scenario = build_scenario_from_config("config/default_sim_config.yaml")
engine = SimulationEngine(scenario=scenario) # Pass scenario during init

# Run the simulation (assuming engine has a `run()` method)
results = engine.run()

# Process results...
print("Simulation finished.")
```

## Directory Structure

/Users/yessine/Helios/
├── config/             # Configuration files (YAML, JSON)
├── data/               # Sample data (scenarios, models)
├── docs/               # Documentation files
├── src/
│   ├── analysis/       # Post-simulation analysis scripts
│   └── helios/         # Main source code for the Helios library
│       ├── cli/        # Command-line interface entry points
│       ├── cognitive/  # Cognitive engine components
│       ├── core/       # Core interfaces and data structures
│       ├── environment/# Environment modeling (propagation, clutter, RCS)
│       ├── hardware/   # Hardware interface abstractions
│       ├── network/    # Network simulation components
│       ├── security/   # Security features (auth, sanitization)
│       ├── simulation/ # Simulation engine implementations
│       ├── utils/      # Utility functions (logging, config loading)
│       └── waveforms/  # Waveform generation and management
├── tests/              # Unit and integration tests
├── Dockerfile          # Docker build definition
├── INTENT.md           # Statement of Intent / Ethical Use Guide
├── LICENSE             # Project license file
├── MANIFEST.in         # Packaging manifest for non-code files
├── README.md           # This file
├── deploy.sh           # Deployment script
├── docker-compose.yml  # Docker Compose configuration
├── pyproject.toml      # Project metadata and dependencies (for Poetry)
└── setup.py            # Python packaging script
```

## Deployment

Helios supports containerized deployment using Docker.

**Build Docker Image:**

```bash
docker build -t helios-rf:latest .
```

**Run Simulation in Docker:**

```bash
docker run -v $(pwd)/config:/app/config -v $(pwd)/data:/data helios-rf:latest \
  helios-sim -c /app/config/default_sim_config.yaml -o /data/results
```

**Using Docker Compose:**

The `docker-compose.yml` file provides configurations for running simulation and analysis services.

```bash
docker-compose up
```

A deployment script (`deploy.sh`) is also available to create a distributable package. 

## Testing

Tests are located in the `/Users/yessine/Helios/tests` directory and can be run using `pytest`.

```bash
# Ensure pytest is installed (usually via Poetry)
poetry run pytest
# Or if pytest is installed globally/in venv
pytest
```

## Contributing

Contributions are welcome! Open an issue on the project repository to discuss changes.

## Ethical Use

The Helios RF Systems Suite is intended for lawful and ethical use in research, development, education, and system analysis. Users are expected to adhere to the principles outlined in the [Statement of Intent (INTENT.md)](/Users/yessine/Helios/INTENT.md) and comply with all applicable laws and regulations regarding RF spectrum usage and system development. Misuse of this software for harmful or illegal activities is strictly prohibited.

## License

This project is licensed under the terms specified in the [LICENSE](/Users/yessine/Helios/LICENSE) file.

## Support

For questions or issues, please open an issue on the project's repository.
```