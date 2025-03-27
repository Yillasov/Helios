import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List

# Define the expected path for results
RESULTS_DIR = '/Users/yessine/Helios/data/results'

def load_simulation_data(run_id: str, file_name: str = 'network_metrics.csv') -> pd.DataFrame:
    """Loads simulation data for a specific run."""
    file_path = os.path.join(RESULTS_DIR, run_id, file_name)
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def analyze_link_quality(df: pd.DataFrame, output_dir: str):
    """Analyzes and plots link quality over time."""
    if df.empty or 'timestamp' not in df.columns or 'link_quality' not in df.columns:
        print("Insufficient data for link quality analysis.")
        return

    plt.figure(figsize=(12, 6))
    for link_id in df['link_id'].unique():
        link_df = df[df['link_id'] == link_id]
        plt.plot(link_df['timestamp'], link_df['link_quality'], label=f'Link {link_id}')

    plt.xlabel("Time (s)")
    plt.ylabel("Link Quality Metric")
    plt.title("Link Quality Over Time")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'link_quality_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved link quality analysis plot to {plot_path}")

def run_analysis(run_id: str):
    """Runs a full analysis suite for a given simulation run."""
    print(f"--- Starting Analysis for Run ID: {run_id} ---")
    run_output_dir = os.path.join(RESULTS_DIR, run_id, 'analysis')
    os.makedirs(run_output_dir, exist_ok=True)

    # Load relevant data files
    network_df = load_simulation_data(run_id, 'network_metrics.csv')
    # Add loading for other data files (e.g., rf_environment.csv, effects.csv)

    # Perform analyses
    analyze_link_quality(network_df, run_output_dir)
    # Add calls to other analysis functions (e.g., analyze_interference, analyze_routing)

    print(f"--- Analysis Complete for Run ID: {run_id} ---")

if __name__ == "__main__":
    # Example usage: Analyze a specific simulation run
    simulation_run_id = "sim_run_example_001" # Replace with actual run ID
    run_analysis(simulation_run_id)