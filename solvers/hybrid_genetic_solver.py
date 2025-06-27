
import time
import random
import logging
from multiprocessing import Pool
from typing import List, Tuple, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from solvers.base_solver import BaseSolver
import numpy as np

def _init_worker(env: PatheryEnv) -> None:
    global solver_env
    solver_env = env

def _calculate_fitness(individual: List[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:

    wall_locations = np.where(solver_env.grid == CellType.WALL.value)
    for y, x in zip(wall_locations[0], wall_locations[1]):
        solver_env.grid[y][x] = CellType.OPEN.value

    for x, y in individual:
        solver_env.step((y,x))

    path = solver_env._calculateShortestPath()
    return len(path), individual


class HybridGeneticSolver(BaseSolver):
    """
    A solver that uses a hybrid genetic algorithm.
    """

    def __init__(self, env: PatheryEnv, population_size: int = 100, num_generations: int = 200, mutation_rate: float = 0.01, elite_size: int = 5, best_known_solution: int = 0, time_limit: Optional[int] = None, **kwargs) -> None:
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
        """
        super().__init__(env, best_known_solution, time_limit)
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
            self._clear_walls()
            self._randomly_place_walls(self.env.wallsToPlace)
            wall_locations = np.where(self.env.grid == CellType.WALL.value)
            population[i] = list(zip(wall_locations[1], wall_locations[0]))


        with Pool(initializer=_init_worker, initargs=(self.env,)) as pool:
            for generation in range(self.num_generations):
                if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                    logging.info(f"Time limit reached. Exiting after {generation} generations.")
                    break
                # Dynamic mutation rate
                current_mutation_rate = max(0.01, self.mutation_rate * (0.95 ** generation))

                logging.info(f"Generation {generation + 1}/{self.num_generations}, Best score so far: {best_path_length}, Mutation rate: {current_mutation_rate:.4f}")
                logging.getLogger().handlers[0].flush()

                # Asynchronously calculate fitness for the population
                results = pool.map(_calculate_fitness, population)

                for score, individual in results:
                    if score > best_path_length:
                        best_path_length = score
                        best_individual = individual
                        if self.best_known_solution > 0 and best_path_length >= self.best_known_solution:
                            logging.info(f"Optimal solution found with length: {best_path_length}. Exiting early.")
                            # Restore the best grid found
                            if best_individual:
                                self._clear_walls()
                                for x, y in best_individual:
                                    self.env.step((y,x))
                            best_path = self.env._calculateShortestPath()
                            return best_path, best_path_length

                # Select parents and carry over elites
                sorted_population = [x for _, x in sorted(zip(results, population), key=lambda pair: pair[0][0], reverse=True)]
                elites = sorted_population[:self.elite_size]
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
            self._clear_walls()
            for x, y in best_individual:
                self.env.step((y,x))

        best_path = self.env._calculateShortestPath()

        return best_path, best_path_length

    def _select_parents(self, population: List[List[Tuple[int, int]]], fitness_scores: List[int], tournament_size: int = 3) -> List[List[Tuple[int, int]]]:
        parents = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def _crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]], num_walls: int) -> List[Tuple[int, int]]:
        if not parent1 or not parent2:
            return parent1 or parent2

        # Single-point crossover
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1) if min(len(parent1), len(parent2)) > 1 else 1
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # Fill remaining genes if necessary
        combined = parent1 + parent2
        for gene in combined:
            if len(child) >= num_walls:
                break
            if gene not in child:
                child.append(gene)

        return child

    def _mutate(self, individual: List[Tuple[int, int]], mutation_rate: float) -> None:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
