
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

logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/solver.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
    parser.add_argument("puzzle_file", help="The path to the puzzle file.")
    parser.add_argument("--solver", type=str, default="memetic", help="The solver to use (hill_climbing, simulated_annealing, hybrid_genetic, memetic).")
    parser.add_argument("--generations", type=int, default=200, help="The number of generations to run for genetic algorithms.")
    parser.add_argument("--restarts", type=int, default=10, help="The number of restarts for the hill climbing solver.")
    parser.add_argument("--cooling_rate", type=float, default=0.003, help="The cooling rate for the simulated annealing solver.")
    args = parser.parse_args()

    # Load the puzzle
    game, best_known_solution = load_puzzle(args.puzzle_file)

    # Create a solver
    if args.solver == "hill_climbing":
        solver = HillClimbingSolver(game, num_restarts=args.restarts)
    elif args.solver == "simulated_annealing":
        solver = SimulatedAnnealingSolver(game, cooling_rate=args.cooling_rate)
    elif args.solver == "hybrid_genetic":
        solver = HybridGeneticSolver(game, num_generations=args.generations, best_known_solution=best_known_solution)
    elif args.solver == "memetic":
        solver = MemeticSolver(game, num_generations=args.generations, best_known_solution=best_known_solution)
    else:
        print(f"Unknown solver: {args.solver}")
        exit(1)

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
