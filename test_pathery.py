
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

class TestPathery(unittest.TestCase):

    def test_simple_path(self) -> None:
        """
        Tests a simple path with a known length.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        path = game.find_path()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_path_with_obstacle(self) -> None:
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

    def test_no_path(self) -> None:
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
            logging.info("".join(row))
            
        final_path = game.find_path()
        self.assertIsNotNone(final_path)
        self.assertEqual(best_path_length, len(final_path))

    def test_checkpoint_path(self) -> None:
        """
        Tests a path with a single checkpoint.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_checkpoint(2, 2, 'A')
        path = game.find_path()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_multiple_checkpoints(self) -> None:
        """
        Tests a path with multiple checkpoints that must be visited in order.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_checkpoint(1, 1, 'A')
        game.add_checkpoint(3, 3, 'B')
        path = game.find_path()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_closest_checkpoint(self) -> None:
        """
        Tests that the path goes to the closest of two same-labeled checkpoints.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_checkpoint(1, 1, 'A')
        game.add_checkpoint(3, 0, 'A')
        path = game.find_path()
        self.assertIsNotNone(path)
        # Path should go to (1,1) then to (4,4), length 9
        # Path to (3,0) is 3, path from there to (4,4) is 5. Total dist 8.
        # Path to (1,1) is 2, path from there to (4,4) is 6. Total dist 8.
        # (1,1) is closer to (0,0) than (3,0).
        self.assertEqual(len(path), 9)

    def test_checkpoint_order(self) -> None:
        """
        Tests that checkpoints are visited in alphabetical order.
        """
        game = PatheryEmulator(6, 6, 0)
        game.set_start(0, 0)
        game.set_finish(5, 5)
        game.add_checkpoint(1, 1, 'B')
        game.add_checkpoint(2, 2, 'C')
        game.add_checkpoint(3, 3, 'A')
        path = game.find_path()
        self.assertIsNotNone(path)
        # Path should be (0,0)->A(3,3)->B(1,1)->C(2,2)->(5,5)
        # (0,0) to A(3,3): dist 6
        # A(3,3) to B(1,1): dist 4
        # B(1,1) to C(2,2): dist 2
        # C(2,2) to (5,5): dist 6
        # Total dist = 6 + 4 + 2 + 6 = 18. Path length = 19.
        self.assertEqual(len(path), 19)

    def test_multiple_instances_of_checkpoint(self) -> None:
        """
        Tests pathfinding with multiple instances of the same checkpoint letters.
        The path should go to the closest 'A', then the closest 'B'.
        """
        game = PatheryEmulator(7, 7, 0)
        game.set_start(0, 0)
        game.set_finish(6, 6)
        game.add_checkpoint(1, 1, 'A') # closer A
        game.add_checkpoint(5, 5, 'A') # further A
        game.add_checkpoint(2, 2, 'B') # closer B from A(1,1)
        game.add_checkpoint(4, 4, 'B') # further B from A(1,1)
        path = game.find_path()
        self.assertIsNotNone(path)
        # Path: start -> closest A (1,1) -> closest B (2,2) -> finish
        # (0,0) to A(1,1): dist 2
        # A(1,1) to B(2,2): dist 2
        # B(2,2) to finish(6,6): dist 8
        # Total dist = 2 + 2 + 8 = 12. Path length = 13.
        self.assertEqual(len(path), 13)

    def test_checkpoint_with_obstacle(self) -> None:
        """
        Tests a path to a checkpoint that is partially obstructed.
        """
        game = PatheryEmulator(5, 5, 0)
        game.set_start(0, 0)
        game.set_finish(4, 4)
        game.add_checkpoint(2, 2, 'A')
        game.add_rock(1, 2)
        game.add_rock(2, 1)
        path = game.find_path()
        self.assertIsNotNone(path)
        # Path to A(2,2) is obstructed.
        # Shortest path from (0,0) to (2,2) with obstacles requires going around.
        # e.g., (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(2,2) is dist 6
        # Path from A(2,2) to finish(4,4) is dist 4.
        # Total dist = 6 + 4 = 10. Path length = 11.
        self.assertEqual(len(path), 11)

if __name__ == '__main__':
    unittest.main()
