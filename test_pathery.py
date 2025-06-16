
import unittest
import logging
from pathery_emulator import PatheryEmulator
from pathery_solver import load_puzzle, solver_factory, load_config
from solvers import (
    HillClimbingSolver,
    SimulatedAnnealingSolver,
    HybridGeneticSolver,
    MemeticSolver,
)

class BaseSolverTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.game, self.best_known_solution = load_puzzle(
            self.config['puzzle_files']['puzzle_1']
        )
        logging.basicConfig(filename=self.config['log_files']['test'], level=logging.INFO, format='%(asctime)s - %(message)s')

    def _test_solver(self, solver_name):
        solver = solver_factory(solver_name, self.game, self.config, self.best_known_solution)
        best_path, best_path_length = solver.solve()
        self.assertIsNotNone(best_path)
        self.assertGreater(best_path_length, 0)

class TestHillClimbingSolver(BaseSolverTest):
    def test_solver(self):
        self._test_solver("hill_climbing")

class TestSimulatedAnnealingSolver(BaseSolverTest):
    def test_solver(self):
        self._test_solver("simulated_annealing")

class TestHybridGeneticSolver(BaseSolverTest):
    def test_solver(self):
        self._test_solver("hybrid_genetic")

class TestMemeticSolver(BaseSolverTest):
    def test_solver(self):
        self._test_solver("memetic")

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
        config = load_config()
        game, _ = load_puzzle(config['puzzle_files']['puzzle_1'])
        
        solver = solver_factory("hybrid_genetic", game, config)
        best_walls, best_path_length = solver.solve()
        
        self.assertIsNotNone(best_walls)
        
        # Log the final grid state
        logging.info("Final grid state:")
        for row in game.grid:
            logging.info("".join(row))
            
        final_path = game.find_path()
        self.assertIsNotNone(final_path)
        self.assertEqual(best_path_length, len(final_path))

if __name__ == '__main__':
    unittest.main()
