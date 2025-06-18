
import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathery_env.envs.pathery import PatheryEnv
from tests.map_builder import MapBuilder

class TestPathery(unittest.TestCase):

    def test_simple_path(self) -> None:
        """
        Tests a simple path with a known length.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)

    def test_path_with_obstacle(self) -> None:
        """
        Tests a path with a simple obstacle.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_rock(2, 2).build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)

    def test_no_path(self) -> None:
        """
        Tests a puzzle with no possible path.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_rock(0, 1).add_rock(1, 0).add_rock(1, 1).build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertEqual(len(path), 0)

    def test_checkpoint_path(self) -> None:
        """
        Tests a path with a single checkpoint.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_checkpoint(2, 2, 'A').build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)

    def test_multiple_checkpoints(self) -> None:
        """
        Tests a path with multiple checkpoints that must be visited in order.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_checkpoint(1, 1, 'A').add_checkpoint(3, 3, 'B').build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)

    def test_closest_checkpoint(self) -> None:
        """
        Tests that the path goes to the closest of two same-labeled checkpoints.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_checkpoint(1, 1, 'A').add_checkpoint(3, 0, 'A').build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 8)

    def test_checkpoint_order(self) -> None:
        """
        Tests that checkpoints are visited in alphabetical order.
        """
        map_string = MapBuilder(6, 6).set_start(0, 0).set_finish(5, 5).add_checkpoint(1, 1, 'B').add_checkpoint(2, 2, 'C').add_checkpoint(3, 3, 'A').build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 18)

    def test_multiple_instances_of_checkpoint(self) -> None:
        """
        Tests pathfinding with multiple instances of the same checkpoint letters.
        The path should go to the closest 'A', then the closest 'B'.
        """
        map_string = MapBuilder(7, 7).set_start(0, 0).set_finish(6, 6).add_checkpoint(1, 1, 'A').add_checkpoint(5, 5, 'A').add_checkpoint(2, 2, 'B').add_checkpoint(4, 4, 'B').build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 12)

    def test_checkpoint_with_obstacle(self) -> None:
        """
        Tests a path to a checkpoint that is partially obstructed.
        """
        map_string = MapBuilder(5, 5).set_start(0, 0).set_finish(4, 4).add_checkpoint(2, 2, 'A').add_rock(1, 2).add_rock(2, 1).build()
        env = PatheryEnv(render_mode=None, map_string=map_string)
        env.reset()
        path = env._calculateShortestPath()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 10)

if __name__ == '__main__':
    unittest.main()
