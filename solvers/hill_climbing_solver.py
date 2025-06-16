
import random
from typing import List, Tuple, Optional
from pathery_emulator import PatheryEmulator
from solvers.base_solver import BaseSolver

class HillClimbingSolver(BaseSolver):
    """
    A solver that uses the hill-climbing algorithm.
    """

    def __init__(self, emulator: PatheryEmulator, num_restarts: int = 10, best_known_solution: int = 0) -> None:
        """
        Initializes the HillClimbingSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
            num_restarts (int): The number of times to restart the algorithm.
        """
        super().__init__(emulator, best_known_solution)
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
            self._randomly_place_walls(self.emulator.num_walls)
            
            _, path_length, _ = self._hill_climb_optimizer(self.emulator.num_walls)

            if path_length > best_path_length:
                best_path_length = path_length
                best_path = self.emulator.find_path()
                best_grid = [row[:] for row in self.emulator.grid]

        # Restore the best grid found
        if best_grid:
            self.emulator.grid = best_grid

        return best_path, best_path_length

    def _hill_climb_optimizer(self, num_walls: int, max_steps: int = 5, num_samples: int = 5) -> Tuple[Optional[List[Tuple[int, int]]], int, List[Tuple[int, int]]]:
        """
        Optimizes a single wall configuration by hill climbing.
        """
        current_path = self.emulator.find_path()
        if not current_path:
            return None, 0, []

        current_path_length = len(current_path)

        for _ in range(max_steps):
            best_neighbor_grid = None
            best_neighbor_path_length = current_path_length

            wall_positions = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == '#':
                        wall_positions.append((x, y))

            # Get all empty squares
            empty_squares = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == ' ':
                        empty_squares.append((x, y))

            if not empty_squares:
                break

            for x_wall, y_wall in wall_positions:
                for x_new, y_new in random.sample(empty_squares, min(len(empty_squares), num_samples)):
                    self.emulator.remove_wall(x_wall, y_wall)
                    self.emulator.add_wall(x_new, y_new)

                    path = self.emulator.find_path()
                    if path and len(path) > best_neighbor_path_length:
                        best_neighbor_path_length = len(path)
                        best_neighbor_grid = [row[:] for row in self.emulator.grid]

                    self.emulator.remove_wall(x_new, y_new)
                    self.emulator.add_wall(x_wall, y_wall)

            if best_neighbor_grid:
                self.emulator.grid = best_neighbor_grid
                current_path_length = best_neighbor_path_length
            else:
                break
        
        final_wall_positions = []
        for y in range(self.emulator.height):
            for x in range(self.emulator.width):
                if self.emulator.grid[y][x] == '#':
                    final_wall_positions.append((x, y))

        return self.emulator.find_path(), current_path_length, final_wall_positions
