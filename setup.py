import os
from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from pyproject.toml
# For a simple approach, we'll hardcode the main dependencies
# In a more sophisticated setup, you might parse pyproject.toml
requirements = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "pyyaml>=6.0",
    "h5py>=3.8.0",
    "numba>=0.57.0",
    "torch>=2.0.0",
    "tensorflow>=2.12.0",
    "scikit-learn>=1.2.0",
    "plotly>=5.14.0",
    "dash>=2.9.0",
]

setup(
    name="helios-rf",
    version="0.1.0",
    author="Helios Team",
    author_email="your.email@example.com",
    description="Advanced RF systems simulation and analysis suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/helios",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: Other/Proprietary License",  # Changed from MIT to proprietary
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "helios": ["config/*.yaml", "config/*.json"],
    },
    entry_points={
        "console_scripts": [
            "helios-sim=helios.cli.simulator:main",
            "helios-analyze=helios.cli.analyzer:main",
        ],
    },
)