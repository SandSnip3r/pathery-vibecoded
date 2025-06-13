import time
import logging
import solver
from pathery_solver import load_puzzle

# Configure logging
logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/benchmark_results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def run_benchmark():
    """
    Runs a single generation of the solver and logs the execution time.
    """
    # Load the puzzle
    puzzle_file = '/usr/local/google/home/victorstone/pathery_project/puzzles/puzzle_2.json'
    game, _ = load_puzzle(puzzle_file)

    # Time a single generation
    start_time = time.time()
    
    solver.solve(
        game.width,
        game.height,
        game.num_walls,
        [rock for rock in game.grid if rock == 'O'],
        game.start,
        game.finish,
        100, 1, 0.01, 5 # Just one generation for the benchmark
    )

    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"Benchmark Result: A single generation took {duration:.4f} seconds.")
    logging.getLogger().handlers[0].flush()
    print(f"Benchmark Result: A single generation took {duration:.4f} seconds.")

if __name__ == '__main__':
    run_benchmark()