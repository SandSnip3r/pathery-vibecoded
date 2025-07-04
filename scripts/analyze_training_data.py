import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
from rich.console import Console
from rich.table import Table

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery_env.envs.pathery import CellType
from src.pathery.utils import load_puzzle


def analyze_data(data_dir: str):
    """
    Analyzes preprocessed data from .pkl files and prints statistics.
    """
    console = Console()
    console.print(f"[bold cyan]Analyzing data from: {data_dir}[/bold cyan]")

    all_rewards = []
    all_wall_counts = []
    all_mutation_types = []
    # all_path_lengths = []

    if not os.path.isdir(data_dir):
        console.print(f"[bold red]Error: Directory not found at {data_dir}[/bold red]")
        return

    filenames = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    if not filenames:
        console.print(f"[bold red]Error: No .pkl files found in {data_dir}[/bold red]")
        return

    # Create a single dummy env to use for calculations
    dummy_env, _ = load_puzzle("ucu/puzzle_6")

    for filename in sorted(filenames):
        console.print(f"Processing {filename}...")
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "rb") as f:
            batch_data = pickle.load(f)
            for entry in batch_data:
                all_rewards.append(entry["reward"])
                state = np.array(entry["pre_mutation_state"])
                wall_count = np.sum(state == CellType.WALL.value)
                all_wall_counts.append(wall_count)
                all_mutation_types.append(entry["mutation_info"]["type"])

                # # Reconstruct the environment state for accurate path calculation
                # env_copy = dummy_env.copy()
                # env_copy.grid = state.astype(np.int32)

                # # Find all start, goal, and checkpoint positions from the grid
                # start_positions = np.argwhere(env_copy.grid == CellType.START.value)
                # goal_positions = np.argwhere(env_copy.grid == CellType.GOAL.value)
                # checkpoints = {}
                # for i in range(len(CellType), dummy_env.cellTypeCount):
                #     if np.any(env_copy.grid == i):
                #         checkpoints[i] = np.argwhere(env_copy.grid == i)

                # # Correctly assign lists of (x, y) tuples
                # env_copy.startPositions = [
                #     (int(pos[1]), int(pos[0])) for pos in start_positions
                # ]
                # env_copy.goalPositions = [
                #     (int(pos[1]), int(pos[0])) for pos in goal_positions
                # ]

                # if checkpoints:
                #     env_copy.checkpoints = {
                #         k: (int(v[0][1]), int(v[0][0])) for k, v in checkpoints.items()
                #     }
                # else:
                #     env_copy.checkpoints = {}
                # env_copy.visited_checkpoints = set()

                # path = env_copy._calculateShortestPath()
                # path_length = len(path) if path is not None and path.any() else 0
                # all_path_lengths.append(path_length)

    console.print("\n[bold green]Analysis Complete![/bold green]")

    # --- Statistics Calculation ---
    def get_stats(data, name):
        data_np = np.array(data)
        q1, median, q3 = np.percentile(data_np, [25, 50, 75])
        return {
            "Metric": name,
            "Min": f"{np.min(data_np):.2f}",
            "Q1": f"{q1:.2f}",
            "Median": f"{median:.2f}",
            "Q3": f"{q3:.2f}",
            "Max": f"{np.max(data_np):.2f}",
            "Mean": f"{np.mean(data_np):.2f}",
            "Std Dev": f"{np.std(data_np):.2f}",
        }

    # --- Display Tables ---
    # Box Plot Stats
    stats_table = Table(title="Box and Whisker Plot Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Min", style="magenta")
    stats_table.add_column("Q1", style="yellow")
    stats_table.add_column("Median", style="green")
    stats_table.add_column("Q3", style="yellow")
    stats_table.add_column("Max", style="magenta")
    stats_table.add_column("Mean", style="blue")
    stats_table.add_column("Std Dev", style="red")

    reward_stats = get_stats(all_rewards, "Reward")
    wall_stats = get_stats(all_wall_counts, "Wall Count")
    # path_length_stats = get_stats(all_path_lengths, "Path Length")

    stats_table.add_row(*reward_stats.values())
    stats_table.add_row(*wall_stats.values())
    # stats_table.add_row(*path_length_stats.values())

    # Mutation Type Stats
    mutation_counts = Counter(all_mutation_types)
    mutation_table = Table(title="Mutation Type Distribution")
    mutation_table.add_column("Mutation Type", style="cyan")
    mutation_table.add_column("Count", style="magenta")
    mutation_table.add_column("Percentage", style="green")

    total_mutations = len(all_mutation_types)
    for m_type, count in mutation_counts.items():
        percentage = (count / total_mutations) * 100
        mutation_table.add_row(m_type, f"{count:,}", f"{percentage:.2f}%")

    console.print(stats_table)
    console.print(mutation_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze preprocessed training data for Pathery."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the preprocessed training data (.pkl files).",
    )
    args = parser.parse_args()
    analyze_data(args.data_dir)
