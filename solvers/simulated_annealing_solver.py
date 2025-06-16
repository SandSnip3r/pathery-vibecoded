
import random
import math
from typing import Tuple, List, Optional
from pathery_emulator import PatheryEmulator
from solvers.base_solver import BaseSolver

class SimulatedAnnealingSolver(BaseSolver):
    """
    A solver that uses the simulated annealing algorithm.
    """

    def __init__(self, emulator: PatheryEmulator, initial_temp: float = 1000, cooling_rate: float = 0.003, best_known_solution: int = 0) -> None:
        """
        Initializes the SimulatedAnnealingSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
            initial_temp (float): The initial temperature.
            cooling_rate (float): The rate at which the temperature cools.
        """
        super().__init__(emulator, best_known_solution)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using simulated annealing.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self._clear_walls()
        self._randomly_place_walls(self.emulator.num_walls)

        current_path = self.emulator.find_path()
        if not current_path:
            return None, 0

        current_path_length = len(current_path)
        best_path = current_path
        best_path_length = current_path_length
        best_grid = [row[:] for row in self.emulator.grid]

        temp = self.initial_temp

        while temp > 1:
            # Create a neighbor by moving a random wall
            wall_positions = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == '#':
                        wall_positions.append((x, y))

            if not wall_positions:
                break

            wall_to_move = random.choice(wall_positions)
            
            while True:
                new_x = random.randint(0, self.emulator.width - 1)
                new_y = random.randint(0, self.emulator.height - 1)
                if self.emulator.grid[new_y][new_x] == ' ':
                    break
            
            self.emulator.remove_wall(wall_to_move[0], wall_to_move[1])
            self.emulator.add_wall(new_x, new_y)

            new_path = self.emulator.find_path()

            if new_path:
                new_path_length = len(new_path)
                
                # If the new solution is better, accept it
                if new_path_length > current_path_length:
                    current_path_length = new_path_length
                    if new_path_length > best_path_length:
                        best_path_length = new_path_length
                        best_path = new_path
                        best_grid = [row[:] for row in self.emulator.grid]
                # If the new solution is worse, accept it with a certain probability
                else:
                    acceptance_probability = math.exp((new_path_length - current_path_length) / temp)
                    if random.random() < acceptance_probability:
                        current_path_length = new_path_length
                    else:
                        # Revert the change
                        self.emulator.remove_wall(new_x, new_y)
                        self.emulator.add_wall(wall_to_move[0], wall_to_move[1])

            # Cool the temperature
            temp *= 1 - self.cooling_rate

        # Restore the best grid found
        if best_grid:
            self.emulator.grid = best_grid

        return best_path, best_path_length
