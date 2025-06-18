
import random

from typing import Tuple, List, Optional
from pathery_env_adapter import PatheryEnvAdapter as PatheryEmulator

class BaseSolver:
    """
    A base class for Pathery solvers.
    """

    def __init__(self, emulator: PatheryEmulator, best_known_solution: int = 0) -> None:
        """
        Initializes the BaseSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
            best_known_solution (int): The best known solution length.
        """
        self.emulator = emulator
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
        for y in range(self.emulator.height):
            for x in range(self.emulator.width):
                if self.emulator.grid[y][x] == '#':
                    self.emulator.remove_wall(x, y)

    def _randomly_place_walls(self, num_walls: int) -> None:
        """
        Randomly places a given number of walls on the emulator's grid.

        Args:
            num_walls (int): The number of walls to place.
        """
        empty_cells = self.emulator.get_empty_cells()
        random.shuffle(empty_cells)
        for i in range(min(num_walls, len(empty_cells))):
            x, y = empty_cells[i]
            self.emulator.add_wall(x, y)
