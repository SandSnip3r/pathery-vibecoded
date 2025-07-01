import unittest

from src.pathery_env.envs.pathery import PatheryEnv
from src.pathery.main import solver_factory
from src.pathery.map_builder import MapBuilder


class BaseSolverTest(unittest.TestCase):
    def setUp(self) -> None:
        map_string = MapBuilder(10, 10, 5).set_start(0, 0).set_finish(9, 9).build()
        self.env = PatheryEnv(render_mode=None, map_string=map_string)
        self.env.reset()

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


class TestSolverBug(BaseSolverTest):
    def test_solver_bug_minimal(self) -> None:
        """
        A minimal test case to reproduce the solver bug.
        """
        solver = solver_factory("hybrid_genetic", self.env)
        best_walls, best_path_length = solver.solve()

        self.assertIsNotNone(best_walls)

        final_path = self.env._calculateShortestPath()
        self.assertIsNotNone(final_path)
        self.assertEqual(best_path_length, len(final_path))


if __name__ == "__main__":
    unittest.main()
