
import json
import logging
import argparse
from pathery_emulator import PatheryEmulator
from solvers import (
    HillClimbingSolver,
    SimulatedAnnealingSolver,
    HybridGeneticSolver,
    MemeticSolver,
)

def load_config(path='config.json'):
    with open(path, 'r') as f:
        return json.load(f)

def solver_factory(solver_name, emulator, config, best_known_solution=0):
    """
    Creates a solver instance from a name and configuration.
    """
    solver_class = {
        "hill_climbing": HillClimbingSolver,
        "simulated_annealing": SimulatedAnnealingSolver,
        "hybrid_genetic": HybridGeneticSolver,
        "memetic": MemeticSolver,
    }.get(solver_name)

    if not solver_class:
        raise ValueError(f"Unknown solver: {solver_name}")

    solver_config = config['solvers'].get(solver_name, {})
    
    # Add best_known_solution to solver_config if it's a relevant parameter
    if "best_known_solution" in solver_class.__init__.__code__.co_varnames:
        solver_config["best_known_solution"] = best_known_solution

    return solver_class(emulator, **solver_config)

def load_puzzle(file_path):
    """
    Loads a puzzle from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        PatheryEmulator: An instance of the PatheryEmulator with the loaded puzzle.
    """
    with open(file_path, 'r') as f:
        puzzle_data = json.load(f)

    game = PatheryEmulator(puzzle_data['width'], puzzle_data['height'], puzzle_data['num_walls'])
    game.set_start(puzzle_data['start'][0], puzzle_data['start'][1])
    game.set_finish(puzzle_data['finish'][0], puzzle_data['finish'][1])

    for rock in puzzle_data['rocks']:
        game.add_rock(rock[0], rock[1])

    return game, puzzle_data['best_solution']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle_name", help="The name of the puzzle to solve (e.g., puzzle_1).")
    parser.add_argument("--solver", type=str, default="memetic", help="The solver to use (hill_climbing, simulated_annealing, hybrid_genetic, memetic).")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Configure logging
    logging.basicConfig(filename=config['log_files']['solver'], level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load the puzzle
    puzzle_file = config['puzzle_files'][args.puzzle_name]
    game, best_known_solution = load_puzzle(puzzle_file)

    # Create a solver
    solver = solver_factory(args.solver, game, config, best_known_solution)

    # Find the best path
    best_path, best_path_length = solver.solve()

    if best_path:
        print(f"Best path found with length: {best_path_length}")
        if best_path_length > best_known_solution:
            print("New best solution found!")
        game.draw_path(best_path)
        game.display()
    else:
        print("No path found.")
