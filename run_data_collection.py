import argparse
import subprocess
from typing import List


def run_in_batches(
    puzzles: List[str], batch_size: int, num_generations: int, data_log_dir: str
):
    """
    Runs the pathery solver for a list of puzzles in batches.
    """
    puzzle_batches = [
        puzzles[i : i + batch_size] for i in range(0, len(puzzles), batch_size)
    ]

    for i, batch in enumerate(puzzle_batches):
        print(f"Running batch {i + 1}/{len(puzzle_batches)}...")
        processes = []
        for puzzle_name in batch:
            command = [
                "python",
                "pathery_solver.py",
                puzzle_name,
                "--solver",
                "genetic",
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
        "--puzzles",
        nargs="+",
        required=True,
        help="The names of the puzzles to solve.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="The number of puzzles to run in each batch.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1000,
        help="Number of generations for the genetic algorithm.",
    )
    parser.add_argument(
        "--data_log_dir",
        type=str,
        default="mutation_logs",
        help="Directory to save mutation data logs.",
    )
    args = parser.parse_args()

    run_in_batches(
        args.puzzles, args.batch_size, args.num_generations, args.data_log_dir
    )
