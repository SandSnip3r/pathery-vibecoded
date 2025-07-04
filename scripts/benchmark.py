import argparse
import json
import logging
import os
import statistics
import sys
import time
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.main import load_config, solver_factory
from src.pathery.utils import load_puzzle


def run_benchmarks(
    puzzles: List[str], solvers: List[str], num_runs: int, num_generations: int
) -> None:
    """
    Runs benchmarks for given puzzles and solvers and generates a performance report.
    """
    config = load_config()
    log_file = os.path.join("output/logs", config["log_files"]["benchmark"])
    logging.basicConfig(
        filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    solver_names = "_".join(solvers)

    # Ensure the benchmarks directory exists
    os.makedirs("benchmarks", exist_ok=True)
    report_filename = f"benchmarks/solver_performance_{solver_names}_{timestamp}.md"

    with open(report_filename, "w") as f:
        f.write("# Solver Performance Report\n\n")
        f.write("## Benchmark Details\n\n")
        f.write(f"- **Solver(s):** {', '.join(solvers)}\n")
        f.write(f"- **Timestamp:** {timestamp}\n")
        f.write(f"- **Runs per Puzzle:** {num_runs}\n\n")

        for solver_name in solvers:
            solver_config = config["solvers"].get(solver_name, {})
            if num_generations:
                solver_config["generations"] = num_generations
            f.write(f"### Configuration for `{solver_name}`\n\n")
            f.write("```json\n")
            f.write(json.dumps(solver_config, indent=2))
            f.write("\n```\n\n")

    for puzzle_name in puzzles:
        with open(report_filename, "a") as f:
            f.write(f"\n## Puzzle: data/puzzles/{puzzle_name}\n\n")
            f.write(
                "| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |\n"
            )
            f.write("|---|---|---|---|---|\n")

        for solver_name in solvers:
            path_lengths = []
            durations = []

            print(f"Running benchmark for {solver_name} on {puzzle_name}...")
            for i in range(num_runs):
                print(f"  Run {i+1}/{num_runs}...")
                game, puzzle_data = load_puzzle(puzzle_name)
                solver_config = config["solvers"].get(solver_name, {})
                if num_generations:
                    solver_config["generations"] = num_generations
                solver = solver_factory(solver_name, game, **solver_config)

                start_time = time.time()
                _, best_path_length = solver.solve()
                end_time = time.time()

                path_lengths.append(best_path_length)
                durations.append(end_time - start_time)

            min_path = min(path_lengths) if path_lengths else 0
            max_path = max(path_lengths) if path_lengths else 0
            mean_path = statistics.mean(path_lengths) if path_lengths else 0
            mean_duration = statistics.mean(durations) if durations else 0

            result = f"| {solver_name} | {min_path} | {max_path} | {mean_path:.2f} | {mean_duration:.4f} |"

            with open(report_filename, "a") as f:
                f.write(result + "\n")

            logging.info(
                f"Benchmark Result for {puzzle_name} with {solver_name}: {result}"
            )
            print(f"Benchmark complete for {solver_name} on {puzzle_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--puzzles",
        nargs="+",
        default=[
            "ucu/puzzle_10",
            "ucu/puzzle_20",
            "ucu/puzzle_30",
            "ucu/puzzle_40",
            "ucu/puzzle_50",
            "ucu/puzzle_60",
            "ucu/puzzle_70",
            "ucu/puzzle_80",
            "ucu/puzzle_90",
            "ucu/puzzle_100",
        ],
        help="The names of the puzzles to solve.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["dqn_genetic"],
        help="The solvers to use.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="The number of times to run the benchmark for each solver.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="The number of generations to run the genetic algorithms for.",
    )
    args = parser.parse_args()
    run_benchmarks(args.puzzles, args.solvers, args.num_runs, args.num_generations)
