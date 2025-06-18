
import json
import logging
import argparse
from typing import Dict, Any, Tuple
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
    parser.add_argument("puzzle_name", help="The name of the puzzle to solve (e.g., puzzle_1).")
    parser.add_argument("--solver", type=str, default="memetic", help="The solver to use (hill_climbing, simulated_annealing, hybrid_genetic, memetic, focused_search).")
    parser.add_argument("--num_generations", type=int, help="Number of generations for genetic algorithms.")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Override config with command-line arguments if provided
    if args.num_generations and args.solver in config['solvers'] and 'num_generations' in config['solvers'][args.solver]:
        config['solvers'][args.solver]['num_generations'] = args.num_generations
    
    # Configure logging
    logging.basicConfig(filename=config['log_files']['solver'], level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load the puzzle
    with open(config['puzzle_files'][args.puzzle_name], 'r') as f:
        puzzle_data = json.load(f)

    builder = MapBuilder(puzzle_data['width'], puzzle_data['height'], puzzle_data['num_walls'])
    builder.set_start(puzzle_data['start'][0], puzzle_data['start'][1])
    builder.set_finish(puzzle_data['finish'][0], puzzle_data['finish'][1])

    for rock in puzzle_data['rocks']:
        builder.add_rock(rock[0], rock[1])

    if 'checkpoints' in puzzle_data:
        for checkpoint in puzzle_data['checkpoints']:
            builder.add_checkpoint(checkpoint[0], checkpoint[1], checkpoint[2])
    
    env = PatheryEnv(render_mode=None, map_string=builder.build())
    env.reset()

    # Create a solver
    solver_config = config['solvers'].get(args.solver, {})
    solver = solver_factory(args.solver, env, **solver_config)

    # Find the best path
    best_path, best_path_length = solver.solve()

    if best_path.any():
        print(f"Best path found with length: {best_path_length}")
        if puzzle_data['best_solution'] == 0 or best_path_length > puzzle_data['best_solution']:
            print("New best solution found!")
        print(env.render())
    else:
        print("No path found.")
