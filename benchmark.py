import time
import logging
from pathery_solver import PatherySolver, load_puzzle

# Configure logging
logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/benchmark_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def run_benchmark():
    """
    Runs the full solver and logs the execution time and final path length.
    """
    # Load the puzzle
    puzzle_file = '/usr/local/google/home/victorstone/pathery_project/puzzles/puzzle_2.json'
    game, _ = load_puzzle(puzzle_file)

    # Create a solver
    solver = PatherySolver(game)

    # Time the full solver
    start_time = time.time()
    
    best_path, best_path_length = solver.solve_hybrid_genetic_algorithm(
        game.num_walls,
        100, # population_size
        50,  # num_generations
        0.1, # mutation_rate
        5    # elite_size
    )

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
