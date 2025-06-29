import random
import math
import time
from typing import Tuple, List, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from solvers.base_solver import BaseSolver
import numpy as np


class SimulatedAnnealingSolver(BaseSolver):
    """
    A solver that uses the simulated annealing algorithm.
    """

    def __init__(
        self,
        env: PatheryEnv,
        initial_temp: float = 1000,
        cooling_rate: float = 0.003,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
    ) -> None:
        """
        Initializes the SimulatedAnnealingSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            initial_temp (float): The initial temperature.
            cooling_rate (float): The rate at which the temperature cools.
        """
        super().__init__(env, best_known_solution, time_limit)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using simulated annealing.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self.env.reset()
        self._randomly_place_walls(self.env.wallsToPlace)

        current_path = self.env._calculateShortestPath()
        current_path_length = len(current_path)
        if not current_path.any():
            return None, 0

        best_path = current_path
        best_path_length = current_path_length
        best_grid = self.env.grid.copy()

        temp = self.initial_temp
        self.start_time = time.time()

        while temp > 1:
            if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                break
            # Create a neighbor by moving a random wall
            wall_positions = np.where(self.env.grid == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))

            if not wall_positions:
                break

            wall_to_move = random.choice(wall_positions)

            empty_cells = np.where(self.env.grid == CellType.OPEN.value)
            empty_cells = list(zip(empty_cells[1], empty_cells[0]))

            if not empty_cells:
                break

            new_x, new_y = random.choice(empty_cells)

            self.env.grid[wall_to_move[1]][wall_to_move[0]] = CellType.OPEN.value
            self.env.grid[new_y][new_x] = CellType.WALL.value

            new_path = self.env._calculateShortestPath()
            new_path_length = len(new_path)

            if new_path.any():
                # If the new solution is better, accept it
                if new_path_length > current_path_length:
                    current_path_length = new_path_length
                    if new_path_length > best_path_length:
                        best_path_length = new_path_length
                        best_path = new_path
                        best_grid = self.env.grid.copy()
                # If the new solution is worse, accept it with a certain probability
                else:
                    acceptance_probability = math.exp(
                        (new_path_length - current_path_length) / temp
                    )
                    if random.random() < acceptance_probability:
                        current_path_length = new_path_length
                    else:
                        # Revert the change
                        self.env.grid[new_y][new_x] = CellType.OPEN.value
                        self.env.grid[wall_to_move[1]][
                            wall_to_move[0]
                        ] = CellType.WALL.value
            else:
                # Revert the change
                self.env.grid[new_y][new_x] = CellType.OPEN.value
                self.env.grid[wall_to_move[1]][wall_to_move[0]] = CellType.WALL.value

            # Cool the temperature
            temp *= 1 - self.cooling_rate

        # Restore the best grid found
        if best_grid is not None:
            self.env.grid = best_grid

        return best_path, best_path_length
