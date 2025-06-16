
import unittest
import logging
from pathery_emulator import PatheryEmulator
from pathery_solver import load_puzzle
from solvers import (
    HillClimbingSolver,
    SimulatedAnnealingSolver,
    HybridGeneticSolver,
    MemeticSolver,
)

logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/test.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class BaseSolverTest(unittest.TestCase):
    def setUp(self):
        self.game, self.best_known_solution = load_puzzle(
            '/usr/local/google/home/victorstone/pathery_project/puzzles/puzzle_1.json'
        )

    def _test_solver(self, solver):
        best_path, best_path_length = solver.solve()
        self.assertIsNotNone(best_path)
        self.assertGreater(best_path_length, 0)

class TestHillClimbingSolver(BaseSolverTest):
    def test_solver(self):
        solver = HillClimbingSolver(self.game)
        self._test_solver(solver)

class TestSimulatedAnnealingSolver(BaseSolverTest):
    def test_solver(self):
        solver = SimulatedAnnealingSolver(self.game)
        self._test_solver(solver)

class TestHybridGeneticSolver(BaseSolverTest):
    def test_solver(self):
        solver = HybridGeneticSolver(self.game, num_generations=5)
        self._test_solver(solver)

class TestMemeticSolver(BaseSolverTest):
    def test_solver(self):
        solver = MemeticSolver(self.game, num_generations=5)
        self._test_solver(solver)

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
        
        solver = HybridGeneticSolver(game, population_size=10, num_generations=5, mutation_rate=0.1, elite_size=2)
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
