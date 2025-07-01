import logging
import time
from typing import Any, List, Optional, Tuple

from pathery_env.envs.pathery import PatheryEnv

from pathery.solvers.base_solver import BaseSolver
from pathery.solvers.hill_climbing_solver import HillClimbingSolver
from pathery.solvers.hybrid_genetic_solver import HybridGeneticSolver


class MemeticSolver(BaseSolver):
    """
    A solver that uses a memetic algorithm.
    """

    def __init__(
        self,
        env: PatheryEnv,
        population_size: int = 100,
        num_generations: int = 200,
        mutation_rate: float = 0.01,
        elite_size: int = 5,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
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
        super().__init__(
            env,
            best_known_solution,
            time_limit,
            kwargs.get("perf_logger"),
        )
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
        if self.perf_logger:
            self.perf_logger.info(f"genetic,{time.time()},start,,,,,,,")
            self.perf_logger.handlers[0].flush()
        genetic_solver = HybridGeneticSolver(
            self.env,
            population_size=self.population_size,
            num_generations=self.num_generations,
            mutation_rate=self.mutation_rate,
            elite_size=self.elite_size,
            best_known_solution=self.best_known_solution,
            time_limit=genetic_time_limit,
            perf_logger=self.perf_logger,
        )
        best_path, _ = genetic_solver.solve()

        if genetic_solver.generations_run == 0:
            logging.warning(
                "The genetic algorithm phase of the memetic solver did not run for any generations. "
                "This is likely due to a short time limit. The initial population generation is time-consuming. "
                "Consider increasing the time limit to allow the genetic algorithm to run."
            )

        # If the genetic algorithm didn't find a solution, start with a random one
        if not best_path.any():
            self.env.reset()
            self._randomly_place_walls(self.env.wallsToPlace)

        if self.perf_logger:
            self.perf_logger.info(f"hill_climbing,{time.time()},start,,,,,,,")
            self.perf_logger.handlers[0].flush()
        hill_climbing_time_limit = (
            self.time_limit - (time.time() - self.start_time)
            if self.time_limit
            else None
        )
        hill_climbing_solver = HillClimbingSolver(
            self.env,
            num_restarts=self.hill_climbing_restarts,
            time_limit=hill_climbing_time_limit,
            perf_logger=self.perf_logger,
        )
        best_path, best_path_length = hill_climbing_solver.solve()

        return best_path, best_path_length
