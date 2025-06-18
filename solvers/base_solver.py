
import random
import numpy as np
from typing import Tuple, List, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType

class BaseSolver:
    """
    A base class for Pathery solvers.
    """

    def __init__(self, env: PatheryEnv, best_known_solution: int = 0) -> None:
        """
        Initializes the BaseSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            best_known_solution (int): The best known solution length.
        """
        self.env = env
        self.best_known_solution = best_known_solution

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _clear_walls(self) -> None:
        """
        Clears all walls from the emulator's grid.
        """
        wall_locations = np.where(self.env.grid == CellType.WALL.value)
        for y, x in zip(wall_locations[0], wall_locations[1]):
            self.env.grid[y][x] = CellType.OPEN.value

    def _randomly_place_walls(self, num_walls: int) -> None:
        """
        Randomly places a given number of walls on the emulator's grid.

        Args:
            num_walls (int): The number of walls to place.
        """
        open_cells = np.where(self.env.grid == CellType.OPEN.value)
        open_cells = list(zip(open_cells[1], open_cells[0]))
        random.shuffle(open_cells)
        for i in range(min(num_walls, len(open_cells))):
            x, y = open_cells[i]
            self.env.step((y, x))
