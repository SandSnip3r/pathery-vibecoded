import argparse
import os
import random
import subprocess
import time
from typing import List


def get_puzzle_files(puzzle_dir: str) -> List[str]:
    """Gets all puzzle files from a directory."""
    return [
        os.path.join(puzzle_dir, f)
        for f in os.listdir(puzzle_dir)
        if f.endswith(".json")
    ]


def run_in_parallel(
    puzzles: List[str],
    min_generations: int,
    max_generations: int,
    data_log_dir: str,
    solver: str,
):
    """
    Runs the pathery solver for a list of puzzles in parallel with random generations.
    """
    processes = []
    for puzzle_name in puzzles:
        num_generations = random.randint(min_generations, max_generations)
        command = [
            "python",
            "src/pathery/main.py",
            puzzle_name,
            "--solver",
            solver,
            "--num_generations",
            str(num_generations),
            "--data_log_dir",
            data_log_dir,
        ]
        processes.append(subprocess.Popen(command))

    for p in processes:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data collection for the Pathery solver."
    )
    parser.add_argument(
        "--puzzle_dir",
        type=str,
        default="data/puzzles/ucu",
        help="Directory containing the puzzles.",
    )
    parser.add_argument(
        "--data_log_dir",
        type=str,
        default="output/dqn_ga_transitions",
        help="Directory to save mutation data logs.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dqn_genetic",
        help="The solver to use for data collection.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration to run the data collection in hours.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="The number of puzzles to run in each batch.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=100,
        help="Minimum number of generations for the genetic algorithm.",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=1000,
        help="Maximum number of generations for the genetic algorithm.",
    )
    args = parser.parse_args()

    puzzle_files = get_puzzle_files(args.puzzle_dir)
    end_time = time.time() + args.duration * 3600

    while time.time() < end_time:
        print("Starting a new batch of puzzles...")
        sampled_puzzles = random.sample(puzzle_files, args.batch_size)
        run_in_parallel(
            sampled_puzzles,
            args.min_generations,
            args.max_generations,
            args.data_log_dir,
            args.solver,
        )
        print("Batch finished.")

    print("Data collection finished.")
