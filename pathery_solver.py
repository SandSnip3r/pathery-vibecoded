
import json
import logging
import argparse
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from pathery_env.envs.pathery import PatheryEnv
from tests.map_builder import MapBuilder
from solvers.base_solver import BaseSolver
from solvers import (
    HillClimbingSolver,
    SimulatedAnnealingSolver,
    HybridGeneticSolver,
    MemeticSolver,
    FocusedSearchSolver,
)

def load_config(path: str = 'config.json') -> Dict[str, Any]:
    with open(path, 'r') as f:
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
    }.get(solver_name)

    if not solver_class:
        raise ValueError(f"Unknown solver: {solver_name}")

    return solver_class(env, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle", help="The name of the puzzle to solve (e.g., puzzle_1) or the path to a puzzle file.")
    parser.add_argument("--solver", type=str, default="memetic", help="The solver to use (hill_climbing, simulated_annealing, hybrid_genetic, memetic, focused_search).")
    parser.add_argument("--num_generations", type=int, help="Number of generations for genetic algorithms.")
    parser.add_argument("--time_limit", type=int, help="Time limit in seconds for the solver.")
    args = parser.parse_args()

    console = Console()

    # Load configuration
    config = load_config()

    # Override config with command-line arguments if provided
    if args.num_generations and args.solver in config['solvers'] and 'num_generations' in config['solvers'][args.solver]:
        config['solvers'][args.solver]['num_generations'] = args.num_generations
    
    # Configure logging
    logging.basicConfig(filename=config['log_files']['solver'], level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load the puzzle
    if args.puzzle in config['puzzle_files']:
        puzzle_path = config['puzzle_files'][args.puzzle]
    else:
        puzzle_path = args.puzzle

    with open(puzzle_path, 'r') as f:
        puzzle_data = json.load(f)

    if 'map_string' in puzzle_data:
        env = PatheryEnv.fromMapString(render_mode="ansi", map_string=puzzle_data['map_string'])
    else:
        builder = MapBuilder(puzzle_data['width'], puzzle_data['height'], puzzle_data['num_walls'])
        builder.set_start(puzzle_data['start'][0], puzzle_data['start'][1])
        builder.set_finish(puzzle_data['finish'][0], puzzle_data['finish'][1])

        for rock in puzzle_data['rocks']:
            builder.add_rock(rock[0], rock[1])

        if 'checkpoints' in puzzle_data:
            for checkpoint in puzzle_data['checkpoints']:
                builder.add_checkpoint(checkpoint[0], checkpoint[1], checkpoint[2])
        
        env = PatheryEnv(render_mode="ansi", map_string=builder.build())
    
    env.reset()

    console.print(Panel(
        f"[bold]Puzzle:[/bold] {args.puzzle}\n"
        f"[bold]Solver:[/bold] {args.solver}\n"
        f"{env.render()}",
        title="[bold cyan]Pathery Puzzle Solver[/bold cyan]"
    ))

    # Create a solver
    solver_config = config['solvers'].get(args.solver, {})
    if args.time_limit:
        solver_config['time_limit'] = args.time_limit
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
            title="[bold green]Solution Found![/bold green]"
        )
        if puzzle_data['best_solution'] == 0 or best_path_length > puzzle_data['best_solution']:
            console.print("[bold yellow]New best solution found![/bold yellow]")
        console.print(solution_panel)
    else:
        console.print(Panel("[bold red]No path found.[/bold red]", title="[bold red]Solver Failed[/bold red]"))
