
import random
from typing import List, Tuple, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from solvers.base_solver import BaseSolver
import numpy as np

class HillClimbingSolver(BaseSolver):
    """
    A solver that uses the hill-climbing algorithm.
    """

    def __init__(self, env: PatheryEnv, num_restarts: int = 10, best_known_solution: int = 0) -> None:
        """
        Initializes the HillClimbingSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            num_restarts (int): The number of times to restart the algorithm.
        """
        super().__init__(env, best_known_solution)
        self.num_restarts = num_restarts

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a hill-climbing algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        best_path = None
        best_path_length = 0
        best_grid = None

        for _ in range(self.num_restarts):
            self._clear_walls()
            self._randomly_place_walls(self.env.wallsToPlace)
            
            _, path_length, _ = self._hill_climb_optimizer(self.env.wallsToPlace)

            if path_length > best_path_length:
                best_path_length = path_length
                best_path = self.env._calculateShortestPath()
                best_grid = self.env.grid.copy()

        # Restore the best grid found
        if best_grid is not None:
            self.env.grid = best_grid

        return best_path, best_path_length

    def _hill_climb_optimizer(self, num_walls: int, max_steps: int = 5, num_samples: int = 5) -> Tuple[Optional[List[Tuple[int, int]]], int, List[Tuple[int, int]]]:
        """
        Optimizes a single wall configuration by hill climbing.
        """
        current_path = self.env._calculateShortestPath()
        current_path_length = len(current_path)
        if not current_path.any():
            return None, 0, []

        for _ in range(max_steps):
            best_neighbor_grid = None
            best_neighbor_path_length = current_path_length

            wall_positions = np.where(self.env.grid == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))

            # Get all empty squares
            empty_squares = np.where(self.env.grid == CellType.OPEN.value)
            empty_squares = list(zip(empty_squares[1], empty_squares[0]))

            if not empty_squares:
                break

            for x_wall, y_wall in wall_positions:
                for x_new, y_new in random.sample(empty_squares, min(len(empty_squares), num_samples)):
                    self.env.grid[y_wall][x_wall] = CellType.OPEN.value
                    self.env.grid[y_new][x_new] = CellType.WALL.value

                    path = self.env._calculateShortestPath()
                    path_length = len(path)
                    if path.any() and path_length > best_neighbor_path_length:
                        best_neighbor_path_length = path_length
                        best_neighbor_grid = self.env.grid.copy()

                    self.env.grid[y_new][x_new] = CellType.OPEN.value
                    self.env.grid[y_wall][x_wall] = CellType.WALL.value

            if best_neighbor_grid is not None:
                self.env.grid = best_neighbor_grid
                current_path_length = best_neighbor_path_length
            else:
                break
        
        final_wall_positions = np.where(self.env.grid == CellType.WALL.value)
        final_wall_positions = list(zip(final_wall_positions[1], final_wall_positions[0]))

        final_path = self.env._calculateShortestPath()
        return final_path, len(final_path), final_wall_positions
