import logging
import random
import time
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
from pathery_env.envs.pathery import CellType, PatheryEnv

from pathery.solvers.base_solver import BaseSolver


def _init_worker(env: PatheryEnv, perf_logger: Optional[logging.Logger] = None) -> None:
    global solver_env
    global worker_perf_logger
    solver_env = env
    worker_perf_logger = perf_logger


def _calculate_fitness_worker(
    individual: List[Tuple[int, int]]
) -> Tuple[int, List[Tuple[int, int]]]:
    """Helper function to calculate fitness in a separate process."""
    fitness, path = HybridGeneticSolver.calculate_fitness_static(individual, solver_env)
    if worker_perf_logger:
        worker_perf_logger.info(f"genetic_worker,{time.time()},,,{fitness},,,,,,")
        worker_perf_logger.handlers[0].flush()
    return fitness, path


class HybridGeneticSolver(BaseSolver):
    """
    A solver that uses a hybrid genetic algorithm.
    """

    @staticmethod
    def calculate_fitness_static(
        individual: List[Tuple[int, int]], env: PatheryEnv
    ) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Calculates the fitness of an individual.
        This is a static method to allow it to be called by the multiprocessing pool.
        """
        env.reset()
        for x, y in individual:
            # Create a temporary BaseSolver to access _add_wall
            BaseSolver(env)._add_wall(x, y)

        path = env._calculateShortestPath()
        if not path.any():
            return -1, individual
        return len(path), individual

    def __init__(
        self,
        env: PatheryEnv,
        population_size: int = 100,
        num_generations: int = 200,
        mutation_rate: float = 0.01,
        elite_size: int = 5,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        perf_logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the HybridGeneticSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            population_size (int): The size of the population in each generation.
            num_generations (int): The number of generations to run.
            mutation_rate (float): The probability of a mutation occurring.
            elite_size (int): The number of top individuals to carry over to the next generation.
            best_known_solution (int): The best known solution length.
            time_limit (Optional[int]): The time limit in seconds for the solver.
            perf_logger (Optional[logging.Logger]): A logger for performance metrics.
        """
        super().__init__(
            env,
            best_known_solution,
            time_limit,
            perf_logger,
        )
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a hybrid genetic algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self.start_time = time.time()
        best_path_length = 0
        best_individual = None

        # Initialize population
        population = [[] for _ in range(self.population_size)]
        for i in range(self.population_size):
            self.env.reset()
            self._randomly_place_walls(self.env.wallsToPlace)
            wall_locations = np.where(self.env.grid == CellType.WALL.value)
            population[i] = list(zip(wall_locations[1], wall_locations[0]))

        with Pool(
            initializer=_init_worker, initargs=(self.env, self.perf_logger)
        ) as pool:
            for generation in range(self.num_generations):
                if (
                    self.time_limit
                    and (time.time() - self.start_time) > self.time_limit
                ):
                    logging.info(
                        f"Time limit reached. Exiting after {generation} generations."
                    )
                    break
                # Dynamic mutation rate
                current_mutation_rate = max(
                    0.01, self.mutation_rate * (0.95**generation)
                )

                logging.info(
                    f"Generation {generation + 1}/{self.num_generations}, Best score so far: {best_path_length}, Mutation rate: {current_mutation_rate:.4f}"
                )
                logging.getLogger().handlers[0].flush()

                # Asynchronously calculate fitness for the population
                results = pool.map(_calculate_fitness_worker, population)

                scores = [r[0] for r in results]
                if self.perf_logger:
                    self.perf_logger.info(
                        f"genetic,{time.time()},{generation},,{np.max(scores)},{best_path_length},{np.mean(scores)},{np.median(scores)},{np.min(scores)},{np.std(scores)}"
                    )
                    self.perf_logger.handlers[0].flush()

                for score, individual in results:
                    if score > best_path_length:
                        best_path_length = score
                        best_individual = individual
                        if (
                            self.best_known_solution > 0
                            and best_path_length >= self.best_known_solution
                        ):
                            logging.info(
                                f"Optimal solution found with length: {best_path_length}. Exiting early."
                            )
                            # Restore the best grid found
                            if best_individual:
                                self.env.reset()
                                for x, y in best_individual:
                                    self._add_wall(x, y)
                            best_path = self.env._calculateShortestPath()
                            return best_path, best_path_length

                # Select parents and carry over elites
                sorted_population = [
                    x
                    for _, x in sorted(
                        zip(results, population),
                        key=lambda pair: pair[0][0],
                        reverse=True,
                    )
                ]
                elites = sorted_population[: self.elite_size]
                parents = self._select_parents(population, [r[0] for r in results])

                # Create new population
                new_population = elites
                for _ in range(self.population_size - self.elite_size):
                    parent1, parent2 = random.choices(parents, k=2)
                    child = self._crossover(parent1, parent2, self.env.wallsToPlace)
                    self._mutate(child, current_mutation_rate)
                    new_population.append(child)

                population = new_population

        # Restore the best grid found
        if best_individual:
            self.env.reset()
            for x, y in best_individual:
                self._add_wall(x, y)

        self.generations_run = generation + 1
        best_path = self.env._calculateShortestPath()

        return best_path, len(best_path)

    def _select_parents(
        self,
        population: List[List[Tuple[int, int]]],
        fitness_scores: List[int],
        tournament_size: int = 3,
    ) -> List[List[Tuple[int, int]]]:
        parents = []
        for _ in range(len(population)):
            tournament = random.sample(
                list(zip(population, fitness_scores)), tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def _crossover(
        self,
        parent1: List[Tuple[int, int]],
        parent2: List[Tuple[int, int]],
        num_walls: int,
    ) -> List[Tuple[int, int]]:
        if not parent1 or not parent2:
            return parent1 or parent2

        # Single-point crossover
        crossover_point = (
            random.randint(1, min(len(parent1), len(parent2)) - 1)
            if min(len(parent1), len(parent2)) > 1
            else 1
        )
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Fill remaining genes if necessary
        combined = parent1 + parent2
        for gene in combined:
            if len(child) >= num_walls:
                break
            if gene not in child:
                child.append(gene)

        return child[:num_walls]

    def _mutate(self, individual: List[Tuple[int, int]], mutation_rate: float) -> None:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
