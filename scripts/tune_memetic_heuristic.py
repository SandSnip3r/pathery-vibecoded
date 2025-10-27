import argparse
import itertools
import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.main import load_config, solver_factory
from src.pathery.utils import load_puzzle


def tune_hyperparameters(
    puzzle_name: str,
    model_path: str,
    num_generations: int,
    q_value_thresholds: list[float],
    local_search_generations: list[int],
):
    """
    Tunes hyperparameters for the memetic_heuristic solver.
    """
    print("Starting hyperparameter tuning for 'memetic_heuristic'...")
    print(f"Puzzle: {puzzle_name}")
    print(f"Generations per run: {num_generations}")
    print("-" * 30)

    config = load_config()
    base_solver_config = config["solvers"].get("memetic_heuristic", {})
    if model_path:
        base_solver_config["model_path"] = model_path
    base_solver_config["generations"] = num_generations

    # Create all combinations of hyperparameters to test
    param_combinations = list(
        itertools.product(q_value_thresholds, local_search_generations)
    )

    results = []

    for i, (q_value_threshold, local_search_generation) in enumerate(
        param_combinations
    ):
        print(
            f"Running combination {i+1}/{len(param_combinations)}: "
            f"q_value_threshold={q_value_threshold}, local_search_generations={local_search_generation}"
        )

        # Create a copy of the base config and update with the current params
        current_config = base_solver_config.copy()
        current_config["q_value_threshold"] = q_value_threshold
        current_config["local_search_generations"] = local_search_generation

        # Load the puzzle and solver
        game, _ = load_puzzle(puzzle_name)
        solver = solver_factory("memetic_heuristic", game, **current_config)

        start_time = time.time()
        _, best_path_length = solver.solve()
        duration = time.time() - start_time

        results.append(
            {
                "q_value_threshold": q_value_threshold,
                "local_search_generations": local_search_generation,
                "best_path": best_path_length,
                "duration": duration,
            }
        )
        print(f"  -> Best Path: {best_path_length}, Duration: {duration:.2f}s")

    print("\n" + "=" * 30)
    print("Tuning Complete!")
    print("=" * 30)

    # Sort results by best path length (descending)
    sorted_results = sorted(results, key=lambda x: x["best_path"], reverse=True)

    print("Top 5 Best Performing Hyperparameter Combinations:")
    for res in sorted_results[:5]:
        print(
            f"  Path: {res['best_path']}, Q-Value Threshold: {res['q_value_threshold']}, "
            f"Local Search Generations: {res['local_search_generations']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune memetic_heuristic solver hyperparameters."
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default="ucu/puzzle_40",
        help="The name of the puzzle to solve for tuning.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/checkpoints_backup_20250704-223429",
        help="Path to the trained DQN model checkpoint.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=100,
        help="The number of generations to run for each tuning trial.",
    )
    parser.add_argument(
        "--q_value_thresholds",
        nargs="*",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="List of q_value_threshold values to test.",
    )
    parser.add_argument(
        "--local_search_generations",
        nargs="*",
        type=int,
        default=[5, 10, 20],
        help="List of local_search_generations values to test.",
    )
    args = parser.parse_args()

    tune_hyperparameters(
        args.puzzle,
        args.model_path,
        args.num_generations,
        args.q_value_thresholds,
        args.local_search_generations,
    )
