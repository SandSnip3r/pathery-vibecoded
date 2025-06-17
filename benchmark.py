from typing import List
import time
import logging
import argparse
import statistics
from pathery_solver import solver_factory, load_puzzle, load_config

def run_benchmarks(puzzles: List[str], solvers: List[str], num_runs: int) -> None:
    """
    Runs benchmarks for given puzzles and solvers and generates a performance report.
    """
    config = load_config()
    log_file = config['log_files']['benchmark']
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    with open("benchmarks/solver_performance.md", "w") as f:
        f.write("# Solver Performance\n\n")
        f.write(f"This document records the performance of the different solvers on all puzzles. The results are based on {num_runs} runs for each solver on each puzzle.\n")

    for puzzle_name in puzzles:
        with open("benchmarks/solver_performance.md", "a") as f:
            f.write(f"\n## {puzzle_name}\n\n")
            f.write("| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |\n")
            f.write("|---|---|---|---|---|\n")

        for solver_name in solvers:
            path_lengths = []
            durations = []

            print(f"Running benchmark for {solver_name} on {puzzle_name}...")
            for i in range(num_runs):
                print(f"  Run {i+1}/{num_runs}...")
                game, best_known_solution = load_puzzle(config['puzzle_files'][puzzle_name])
                solver = solver_factory(solver_name, game, config, best_known_solution)

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
            
            with open("benchmarks/solver_performance.md", "a") as f:
                f.write(result + "\n")
            
            logging.info(f"Benchmark Result for {puzzle_name} with {solver_name}: {result}")
            print(f"Benchmark complete for {solver_name} on {puzzle_name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzles", nargs='+', default=['puzzle_1', 'puzzle_2', 'puzzle_3', 'puzzle_4', 'puzzle_5'], help="The names of the puzzles to solve.")
    parser.add_argument("--solvers", nargs='+', default=['hill_climbing', 'simulated_annealing', 'hybrid_genetic', 'memetic'], help="The solvers to use.")
    parser.add_argument("--num_runs", type=int, default=5, help="The number of times to run the benchmark for each solver.")
    args = parser.parse_args()
    run_benchmarks(args.puzzles, args.solvers, args.num_runs)
