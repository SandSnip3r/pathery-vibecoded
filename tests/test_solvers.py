import unittest
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathery_env.envs.pathery import PatheryEnv
from tests.map_builder import MapBuilder
from pathery_solver import solver_factory


class BaseSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        map_string = MapBuilder(10, 10, 5).set_start(0, 0).set_finish(9, 9).build()
        self.env = PatheryEnv(render_mode=None, map_string=map_string)
        self.env.reset()
        logging.basicConfig(
            filename="test.log", level=logging.INFO, format="%(asctime)s - %(message)s"
        )

    def _test_solver(self, solver_name: str) -> None:
        solver = solver_factory(solver_name, self.env)
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
        map_string = MapBuilder(10, 10, 5).set_start(0, 0).set_finish(9, 9).build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()

        solver = solver_factory("hybrid_genetic", env)
        best_walls, best_path_length = solver.solve()

        self.assertIsNotNone(best_walls)

        # Log the final grid state
        logging.info("Final grid state:")
        logging.info(env.render())

        final_path = env._calculateShortestPath()
        self.assertIsNotNone(final_path)
        self.assertEqual(best_path_length, len(final_path))


if __name__ == "__main__":
    unittest.main()
