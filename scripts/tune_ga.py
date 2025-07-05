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
    epsilons: list[float],
    mutation_rates: list[float],
):
    """
    Tunes hyperparameters for the dqn_genetic solver.
    """
    print("Starting hyperparameter tuning...")
    print(f"Puzzle: {puzzle_name}")
    print(f"Model: {model_path}")
    print(f"Generations per run: {num_generations}")
    print("-" * 30)

    config = load_config()
    base_solver_config = config["solvers"].get("dqn_genetic", {})
    base_solver_config["model_path"] = model_path
    base_solver_config["generations"] = num_generations

    # Create all combinations of hyperparameters to test
    param_combinations = list(itertools.product(epsilons, mutation_rates))

    results = []

    for i, (epsilon, mutation_rate) in enumerate(param_combinations):
        print(
            f"Running combination {i+1}/{len(param_combinations)}: "
            f"epsilon={epsilon}, mutation_rate={mutation_rate}"
        )

        # Create a copy of the base config and update with the current params
        current_config = base_solver_config.copy()
        current_config["epsilon"] = epsilon
        current_config["mutation_rate"] = mutation_rate

        # Load the puzzle and solver
        game, _ = load_puzzle(puzzle_name)
        solver = solver_factory("dqn_genetic", game, **current_config)

        start_time = time.time()
        _, best_path_length = solver.solve()
        duration = time.time() - start_time

        print(f"  -> Best Path: {best_path_length}, Duration: {duration:.2f}s")
        results.append(
            {
                "epsilon": epsilon,
                "mutation_rate": mutation_rate,
                "best_path": best_path_length,
                "duration": duration,
            }
        )

    print("\n" + "=" * 30)
    print("Tuning Complete!")
    print("=" * 30)

    # Sort results by best path length (descending)
    sorted_results = sorted(results, key=lambda x: x["best_path"], reverse=True)

    print("Top 5 Best Performing Hyperparameter Combinations:")
    for res in sorted_results[:5]:
        print(
            f"  Path: {res['best_path']}, Epsilon: {res['epsilon']}, "
            f"Mutation Rate: {res['mutation_rate']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune dqn_genetic solver hyperparameters."
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default="ucu/puzzle_10",
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
        "--epsilons",
        nargs="*",
        type=float,
        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0],
        help="List of epsilon values to test.",
    )
    parser.add_argument(
        "--mutation_rates",
        nargs="*",
        type=float,
        default=[0.05, 0.1, 0.2, 0.5, 0.8],
        help="List of mutation rate values to test.",
    )
    args = parser.parse_args()

    tune_hyperparameters(
        args.puzzle,
        args.model_path,
        args.num_generations,
        args.epsilons,
        args.mutation_rates,
    )
