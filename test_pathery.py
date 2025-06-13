
import unittest
import logging
from pathery_emulator import PatheryEmulator
from pathery_solver import PatherySolver, load_puzzle

logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/test.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class TestPathery(unittest.TestCase):

    def test_simple_path(self):
        """
        Tests a simple path with a known length.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        path = game.find_path()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_path_with_obstacle(self):
        """
        Tests a path with a simple obstacle.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_rock(2, 2)
        path = game.find_path()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_no_path(self):
        """
        Tests a puzzle with no possible path.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_rock(0, 1)
        game.add_rock(1, 0)
        game.add_rock(1, 1)
        path = game.find_path()
        self.assertIsNone(path)

    def test_solver_bug_minimal(self):
        """
        A minimal test case to reproduce the solver bug.
        """
        game, _ = load_puzzle('/usr/local/google/home/victorstone/pathery_project/puzzles/puzzle_1.json')
        
        solver = PatherySolver(game)
        best_path, best_path_length = solver.solve_hybrid_genetic_algorithm(game.num_walls, 10, 5, 0.1, 2)
        
        self.assertIsNotNone(best_path)
        
        # Log the final grid state
        logging.info("Final grid state:")
        for row in game.grid:
            logging.info("".join(row))
            
        final_path = game.find_path()
        self.assertIsNotNone(final_path)
        self.assertEqual(len(best_path), len(final_path))

if __name__ == '__main__':
    unittest.main()
