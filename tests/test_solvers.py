
import unittest
import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathery_env_adapter import PatheryEnvAdapter as PatheryEmulator
from pathery_solver import load_puzzle, solver_factory, load_config
from solvers import (
    HillClimbingSolver,
    SimulatedAnnealingSolver,
    HybridGeneticSolver,
    MemeticSolver,
)

class BaseSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.game, self.best_known_solution = load_puzzle(
            self.config['puzzle_files']['puzzle_1']
        )
        logging.basicConfig(filename=self.config['log_files']['test'], level=logging.INFO, format='%(asctime)s - %(message)s')

    def _test_solver(self, solver_name: str) -> None:
        solver = solver_factory(solver_name, self.game, self.config, self.best_known_solution)
        best_path, best_path_length = solver.solve()
        self.assertIsNotNone(best_path)
        self.assertGreater(best_path_length, 0)

class TestHillClimbingSolver(BaseSolverTest):
    def test_solver(self) -> None:
        self._test_solver("hill_climbing")

class TestSimulatedAnnealingSolver(BaseSolverTest):
    def test_solver(self) -> None:
        self._test_solver("simulated_annealing")

class TestHybridGeneticSolver(BaseSolverTest):
    def test_solver(self) -> None:
        self._test_solver("hybrid_genetic")

class TestMemeticSolver(BaseSolverTest):
    def test_solver(self) -> None:
        self._test_solver("memetic")

class TestSolverBug(unittest.TestCase):
    def test_solver_bug_minimal(self) -> None:
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
            logging.info("".join(map(str, row)))
            
        final_path, final_path_length = game.find_path()
        self.assertIsNotNone(final_path)
        self.assertEqual(best_path_length, final_path_length)

if __name__ == '__main__':
    unittest.main()
