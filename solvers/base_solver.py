
import random

class BaseSolver:
    """
    A base class for Pathery solvers.
    """

    def __init__(self, emulator):
        """
        Initializes the BaseSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
        """
        self.emulator = emulator

    def solve(self):
        """
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _clear_walls(self):
        """
        Clears all walls from the emulator's grid.
        """
        for y in range(self.emulator.height):
            for x in range(self.emulator.width):
                if self.emulator.grid[y][x] == '#':
                    self.emulator.remove_wall(x, y)

    def _randomly_place_walls(self, num_walls):
        """
        Randomly places a given number of walls on the emulator's grid.

        Args:
            num_walls (int): The number of walls to place.
        """
        for _ in range(num_walls):
            while True:
                x = random.randint(0, self.emulator.width - 1)
                y = random.randint(0, self.emulator.height - 1)
                if self.emulator.grid[y][x] == ' ':
                    self.emulator.add_wall(x, y)
                    break
