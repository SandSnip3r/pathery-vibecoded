
import time
from typing import Tuple, List, Optional, Any
from pathery_env.envs.pathery import PatheryEnv, CellType
from solvers.base_solver import BaseSolver
from solvers.hybrid_genetic_solver import HybridGeneticSolver
from solvers.hill_climbing_solver import HillClimbingSolver
import numpy as np

class MemeticSolver(BaseSolver):
    """
    A solver that uses a memetic algorithm.
    """

    def __init__(self, env: PatheryEnv, population_size: int = 100, num_generations: int = 200, mutation_rate: float = 0.01, elite_size: int = 5, best_known_solution: int = 0, time_limit: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initializes the MemeticSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            population_size (int): The size of the population in each generation.
            num_generations (int): The number of generations to run.
            mutation_rate (float): The probability of a mutation occurring.
            elite_size (int): The number of top individuals to carry over to the next generation.
            best_known_solution (int): The best known solution length.
            time_limit (Optional[int]): The time limit in seconds for the solver.
        """
        super().__init__(env, best_known_solution, time_limit)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.hill_climbing_restarts = kwargs.get("hill_climbing_restarts", 5)

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a memetic algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self.start_time = time.time()

        genetic_time_limit = self.time_limit * 0.8 if self.time_limit else None

        # First, run the genetic algorithm to find a good starting solution
        genetic_solver = HybridGeneticSolver(
            self.env,
            self.population_size,
            self.num_generations,
            self.mutation_rate,
            self.elite_size,
            self.best_known_solution,
            time_limit=genetic_time_limit
        )
        best_path, _ = genetic_solver.solve()

        # If the genetic algorithm didn't find a solution, start with a random one
        if not best_path.any():
            self._clear_walls()
            self._randomly_place_walls(self.env.wallsToPlace)

        hill_climbing_time_limit = self.time_limit - (time.time() - self.start_time) if self.time_limit else None
        hill_climbing_solver = HillClimbingSolver(self.env, num_restarts=self.hill_climbing_restarts, time_limit=hill_climbing_time_limit)
        best_path, best_path_length, final_walls = hill_climbing_solver._hill_climb_optimizer(self.env.wallsToPlace)

        self._clear_walls()
        self.env.remainingWalls = self.env.wallsToPlace
        for x, y in final_walls:
            self.env.step((y,x))

        return best_path, best_path_length
