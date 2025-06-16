import time
import logging
from pathery_solver import solver_factory, load_puzzle, load_config

def run_benchmark():
    """
    Runs the full solver and logs the execution time and final path length.
    """
    # Load configuration
    config = load_config()

    # Configure logging
    logging.basicConfig(filename=config['log_files']['benchmark'], level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load the puzzle
    puzzle_file = config['puzzle_files']['puzzle_2']
    game, best_known_solution = load_puzzle(puzzle_file)

    # Create a solver
    solver = solver_factory("memetic", game, config, best_known_solution)

    # Time the full solver
    start_time = time.time()
    
    best_path, best_path_length = solver.solve()

    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"Benchmark Result: Final path length of {best_path_length} in {duration:.4f} seconds.")
    logging.getLogger().handlers[0].flush()
    print(f"Benchmark Result: Final path length of {best_path_length} in {duration:.4f} seconds.")

    if best_path:
        print("Solution found:")
        game.draw_path(best_path)
        game.display()
    else:
        print("No path found.")

if __name__ == '__main__':
    run_benchmark()
