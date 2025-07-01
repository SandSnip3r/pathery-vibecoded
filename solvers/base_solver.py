import logging
import random
from typing import Optional

import numpy as np
from pathery_env.envs.pathery import CellType, PatheryEnv


class BaseSolver:
    """
    A base class for Pathery solvers.
    """

    def __init__(
        self,
        env: PatheryEnv,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        perf_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the BaseSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            best_known_solution (int): The best known solution length.
            time_limit (Optional[int]): The time limit in seconds for the solver.
            perf_logger (Optional[logging.Logger]): A logger for performance metrics.
        """
        self.env = env
        self.best_known_solution = best_known_solution
        self.time_limit = time_limit
        self.perf_logger = perf_logger
        self.start_time = None

    def _add_wall(self, x: int, y: int) -> None:
        """
        Adds a wall to the grid and decrements the remaining walls count.
        """
        self.env.step((y, x))

    def _remove_wall(self, x: int, y: int) -> None:
        """
        Removes a wall from the grid and increments the remaining walls count.
        """
        self.env.grid[y][x] = CellType.OPEN.value
        self.env.remainingWalls += 1

    def _randomly_place_walls(self, num_walls: int) -> None:
        """
        Randomly places a given number of walls on the emulator's grid.

        Args:
            num_walls (int): The number of walls to place.
        """
        open_cells = np.where(self.env.grid == CellType.OPEN.value)
        open_cells = list(zip(open_cells[1], open_cells[0]))
        random.shuffle(open_cells)

        walls_placed = 0
        for x, y in open_cells:
            if walls_placed >= num_walls:
                break

            self._add_wall(x, y)
            path = self.env._calculateShortestPath()

            if path is not None and path.any():
                walls_placed += 1
            else:
                self._remove_wall(x, y)
