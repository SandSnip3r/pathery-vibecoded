import argparse
import json
import logging
import os
from typing import Any, Dict

from pathery_env.envs.pathery import PatheryEnv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from solvers import (
    DqnGeneticSolver,
    FocusedSearchSolver,
    GeneticSolver,
    HillClimbingSolver,
    HybridGASolver,
    HybridGeneticSolver,
    MemeticSolver,
    SimulatedAnnealingSolver,
)
from solvers.base_solver import BaseSolver
from utils import load_puzzle


def load_config(path: str = "config.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def solver_factory(solver_name: str, env: PatheryEnv, **kwargs) -> BaseSolver:
    """
    Creates a solver instance from a name and configuration.
    """
    solver_class = {
        "hill_climbing": HillClimbingSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
        "hybrid_genetic": HybridGeneticSolver,
        "memetic": MemeticSolver,
        "focused_search": FocusedSearchSolver,
        "genetic": GeneticSolver,
        "hybrid_ga": HybridGASolver,
        "dqn_genetic": DqnGeneticSolver,
    }.get(solver_name)

    if not solver_class:
        raise ValueError(f"Unknown solver: {solver_name}")

    return solver_class(env, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "puzzle",
        help="The name of the puzzle to solve (e.g., puzzle_1) or the path to a puzzle file.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="memetic",
        help="The solver to use (hill_climbing, simulated_annealing, hybrid_genetic, memetic, focused_search, genetic).",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        help="Number of generations for genetic algorithms.",
    )
    parser.add_argument(
        "--time_limit", type=int, help="Time limit in seconds for the solver."
    )
    parser.add_argument(
        "--perf_log_file",
        type=str,
        help="Path to a file to write structured performance logs (CSV format).",
    )
    parser.add_argument(
        "--data_log_dir",
        type=str,
        help="Path to a directory to write mutation data logs (JSONL format).",
    )
    args = parser.parse_args()

    console = Console()

    # Load configuration
    config = load_config()

    # Override config with command-line arguments if provided
    if (
        args.num_generations
        and args.solver in config["solvers"]
        and "num_generations" in config["solvers"][args.solver]
    ):
        config["solvers"][args.solver]["num_generations"] = args.num_generations

    # Configure general application logging
    log_file = os.path.join("logs", config["log_files"]["solver"])
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Configure dedicated performance logger if requested
    perf_logger = None
    if args.perf_log_file:
        perf_logger = logging.getLogger("performance")
        perf_logger.setLevel(logging.INFO)
        # Prevent perf_logger from propagating to the root logger
        perf_logger.propagate = False
        # Overwrite the file if it exists
        handler = logging.FileHandler(args.perf_log_file, mode="w")
        handler.setFormatter(logging.Formatter("%(message)s"))
        perf_logger.addHandler(handler)
        # Write the unified header
        perf_logger.info(
            "phase,timestamp,generation,step,score,best_so_far,mean,median,min,std_dev"
        )

    # Load the puzzle
    env, puzzle_data = load_puzzle(args.puzzle)

    console.print(
        Panel(
            f"[bold]Puzzle:[/bold] {args.puzzle}\n"
            f"[bold]Solver:[/bold] {args.solver}\n"
            f"{env.render()}",
            title="[bold cyan]Pathery Puzzle Solver[/bold cyan]",
        )
    )

    # Create a solver
    solver_config = config["solvers"].get(args.solver, {})
    if args.time_limit:
        solver_config["time_limit"] = args.time_limit
    if args.num_generations:
        solver_config["generations"] = args.num_generations
    if args.data_log_dir:
        solver_config["data_log_dir"] = args.data_log_dir
    # Pass the performance logger to the solver
    solver_config["perf_logger"] = perf_logger
    solver = solver_factory(args.solver, env, **solver_config)

    # Find the best path
    with Progress(
        "[progress.description]{task.description}",
        transient=True,
    ) as progress:
        progress.add_task("Solving puzzle...", total=None)
        best_path, best_path_length = solver.solve()

    if best_path is not None and best_path.any():
        solution_panel = Panel(
            f"[bold]Best path found with length:[/bold] {best_path_length}\n"
            f"{env.render()}",
            title="[bold green]Solution Found![/bold green]",
        )
        if (
            puzzle_data["best_solution"] == 0
            or best_path_length > puzzle_data["best_solution"]
        ):
            console.print("[bold yellow]New best solution found![/bold yellow]")
        console.print(solution_panel)
    else:
        console.print(
            Panel(
                "[bold red]No path found.[/bold red]",
                title="[bold red]Solver Failed[/bold red]",
            )
        )
