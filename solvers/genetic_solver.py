import time
import random
import logging
import json
import os
from typing import List, Tuple, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from solvers.base_solver import BaseSolver
import numpy as np


class GeneticSolver(BaseSolver):
    """
    A solver that uses a genetic algorithm.
    """

    def __init__(
        self,
        env: PatheryEnv,
        population_size: int = 100,
        generations: int = 100,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        perf_logger: Optional[logging.Logger] = None,
        data_log_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the GeneticSolver.
        """
        super().__init__(
            env,
            best_known_solution,
            time_limit,
            perf_logger,
        )
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.data_logger = None
        if data_log_dir:
            log_file = os.path.join(data_log_dir, f"mutations_{os.getpid()}.jsonl")
            self.data_logger = open(log_file, "w")

    def __del__(self):
        if self.data_logger:
            self.data_logger.close()

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a genetic algorithm.
        """
        self.start_time = time.time()
        self._initialize_population()
        best_solution = None
        best_fitness = 0

        for gen in range(self.generations):
            if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                print(f"Time limit reached. Exiting after {gen} generations.")
                break

            # Evaluate fitness of population
            fitness_scores = [
                self._calculate_fitness(chromosome) for chromosome in self.population
            ]

            # Find best solution in current generation
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_solution_index = fitness_scores.index(max_fitness)
                best_solution = self.population[best_solution_index]

            if self.perf_logger:
                self.perf_logger.info(f"genetic,{time.time()},{gen},0,{max_fitness},,,")
                self.perf_logger.handlers[0].flush()

            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)

                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._two_point_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                if random.random() < self.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutate(offspring2)

                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = new_population

        # Set the environment to the best solution found
        if best_solution is not None:
            self.env.grid = best_solution
            best_path = self.env._calculateShortestPath()
            return best_path, len(best_path)
        else:
            return None, 0

    def _initialize_population(self):
        """
        Initializes the population of solutions.
        """
        self.population = []
        for _ in range(self.population_size):
            self.env.reset()
            self._randomly_place_walls(self.env.wallsToPlace)
            self.population.append(self.env.grid.copy())

    def _calculate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Calculates the fitness of a chromosome.
        """
        original_grid = self.env.grid.copy()
        self.env.grid = chromosome
        path = self.env._calculateShortestPath()
        self.env.grid = original_grid  # Restore original grid
        if path is not None and path.any():
            return float(len(path))
        return 0.0

    def _tournament_selection(self, fitness_scores: List[float]) -> np.ndarray:
        """
        Selects a parent using tournament selection.
        """
        tournament = random.sample(
            list(enumerate(fitness_scores)), self.tournament_size
        )
        winner_index, _ = max(tournament, key=lambda item: item[1])
        return self.population[winner_index]

    def _two_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs two-point crossover on two parents.
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        height, width = self.env.grid.shape
        x1, x2 = sorted(random.sample(range(width), 2))
        y1, y2 = sorted(random.sample(range(height), 2))

        # Swap the rectangular region
        offspring1[y1:y2, x1:x2] = parent2[y1:y2, x1:x2]
        offspring2[y1:y2, x1:x2] = parent1[y1:y2, x1:x2]

        return offspring1, offspring2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Performs a random mutation on a chromosome.
        """
        pre_mutation_state = chromosome.copy()
        fitness_before = self._calculate_fitness(pre_mutation_state)

        mutated_chromosome = chromosome.copy()
        mutation_type = random.choice(["MOVE", "ADD", "REMOVE"])

        if mutation_type == "MOVE":
            wall_positions = np.where(mutated_chromosome == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))
            empty_squares = np.where(mutated_chromosome == CellType.OPEN.value)
            empty_squares = list(zip(empty_squares[1], empty_squares[0]))

            if not wall_positions or not empty_squares:
                return mutated_chromosome  # No possible mutation

            wall_to_move = random.choice(wall_positions)
            new_position = random.choice(empty_squares)

            mutated_chromosome[wall_to_move[1], wall_to_move[0]] = CellType.OPEN.value
            mutated_chromosome[new_position[1], new_position[0]] = CellType.WALL.value
            mutation_info = {
                "type": "MOVE",
                "from": [int(wall_to_move[0]), int(wall_to_move[1])],
                "to": [int(new_position[0]), int(new_position[1])],
            }

        elif mutation_type == "ADD":
            empty_squares = np.where(mutated_chromosome == CellType.OPEN.value)
            empty_squares = list(zip(empty_squares[1], empty_squares[0]))

            if not empty_squares:
                return mutated_chromosome  # No possible mutation

            new_wall_position = random.choice(empty_squares)
            mutated_chromosome[new_wall_position[1], new_wall_position[0]] = (
                CellType.WALL.value
            )
            mutation_info = {
                "type": "ADD",
                "to": [int(new_wall_position[0]), int(new_wall_position[1])],
            }

        elif mutation_type == "REMOVE":
            wall_positions = np.where(mutated_chromosome == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))

            if not wall_positions:
                return mutated_chromosome  # No possible mutation

            wall_to_remove = random.choice(wall_positions)
            mutated_chromosome[wall_to_remove[1], wall_to_remove[0]] = (
                CellType.OPEN.value
            )
            mutation_info = {
                "type": "REMOVE",
                "from": [int(wall_to_remove[0]), int(wall_to_remove[1])],
            }

        fitness_after = self._calculate_fitness(mutated_chromosome)
        reward = fitness_after - fitness_before

        if self.data_logger:
            log_entry = {
                "pre_mutation_state": pre_mutation_state.tolist(),
                "mutation_info": mutation_info,
                "reward": reward,
            }
            self.data_logger.write(json.dumps(log_entry) + "\n")

        return mutated_chromosome
