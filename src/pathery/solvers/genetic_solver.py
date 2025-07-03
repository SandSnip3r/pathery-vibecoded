import json
import logging
import os
import random
import time
from typing import IO, List, Optional, Tuple
from pathery_env.envs.pathery import PatheryEnv, CellType
from pathery.solvers.base_solver import BaseSolver
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
        elitism_size: int = 2,
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
        self.elitism_size = elitism_size
        self.population = []
        self.data_log_dir = data_log_dir

    def __del__(self):
        pass

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a genetic algorithm.
        """
        self.start_time = time.time()
        self._initialize_population()
        best_solution = None
        best_fitness = 0

        data_logger = None
        if self.data_log_dir:
            log_file = os.path.join(self.data_log_dir, f"mutations_{os.getpid()}.jsonl")
            try:
                data_logger = open(log_file, "w")
            except IOError as e:
                print(f"Error opening data log file: {e}")
                # Continue without logging if the file can't be opened.
                self.data_log_dir = None

        try:
            for gen in range(self.generations):
                if (
                    self.time_limit
                    and (time.time() - self.start_time) > self.time_limit
                ):
                    print(f"Time limit reached. Exiting after {gen} generations.")
                    break

                # Evaluate fitness of population
                fitness_scores = [
                    self._calculate_fitness(env) for env in self.population
                ]

                # Find best solution in current generation
                max_fitness = max(fitness_scores)
                if max_fitness > best_fitness:
                    best_fitness = max_fitness
                    best_solution_index = fitness_scores.index(max_fitness)
                    best_solution = self.population[best_solution_index]

                if self.perf_logger:
                    self.perf_logger.info(
                        f"genetic,{time.time()},{gen},0,{max_fitness},,,,"
                    )
                    self.perf_logger.handlers[0].flush()

                self._before_selection_hook(gen)

                # Create new population
                new_population = []

                # Elitism: Carry over the best individuals
                elite_indices = sorted(
                    range(len(fitness_scores)),
                    key=lambda i: fitness_scores[i],
                    reverse=True,
                )[: self.elitism_size]
                for i in elite_indices:
                    new_population.append(self.population[i])

                mutations = 0
                while len(new_population) < self.population_size:
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)

                    if random.random() < self.crossover_rate:
                        offspring1, offspring2 = self._two_point_crossover(
                            parent1, parent2
                        )
                    else:
                        offspring1, offspring2 = parent1.copy(), parent2.copy()

                    if random.random() < self.mutation_rate:
                        offspring1 = self._mutate(offspring1, data_logger)
                        mutations += 1
                    if random.random() < self.mutation_rate:
                        offspring2 = self._mutate(offspring2, data_logger)
                        mutations += 1

                    new_population.append(offspring1)
                    new_population.append(offspring2)

                self.population = new_population
                self._after_new_population_hook(gen)
        finally:
            if data_logger:
                data_logger.close()

        # Set the environment to the best solution found
        if best_solution is not None:
            self.env.grid = best_solution.grid
            self.env.remainingWalls = best_solution.remainingWalls
            best_path = self.env._calculateShortestPath()
            return best_path, len(best_path)
        else:
            return None, 0

    def _initialize_population(self):
        """
        Initializes the population of solutions.
        """
        self.population = []
        for i in range(self.population_size):
            env_copy = self.env.copy()
            open_cells = np.where(env_copy.grid == CellType.OPEN.value)
            num_open_cells = len(open_cells[0])
            max_walls = min(env_copy.wallsToPlace, num_open_cells)
            num_walls = random.randint(1, max_walls)
            BaseSolver(env_copy)._randomly_place_walls(num_walls)
            self.population.append(env_copy)

    def _calculate_fitness(self, env: PatheryEnv) -> float:
        """
        Calculates the fitness of a chromosome.
        """
        path = env._calculateShortestPath()
        if path is not None and path.any():
            return float(len(path))
        return 0.0

    def _tournament_selection(self, fitness_scores: List[float]) -> PatheryEnv:
        """
        Selects a parent using tournament selection.
        """
        tournament = random.sample(
            list(enumerate(fitness_scores)), self.tournament_size
        )
        winner_index, _ = max(tournament, key=lambda item: item[1])
        return self.population[winner_index]

    def _two_point_crossover(
        self, parent1: PatheryEnv, parent2: PatheryEnv
    ) -> Tuple[PatheryEnv, PatheryEnv]:
        """
        Performs two-point crossover on two parents.
        """
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        height, width = self.env.grid.shape
        x1, x2 = sorted(random.sample(range(width), 2))
        y1, y2 = sorted(random.sample(range(height), 2))

        crossover_coords = []
        for y in range(y1, y2):
            for x in range(x1, x2):
                crossover_coords.append((x, y))
        random.shuffle(crossover_coords)

        for x, y in crossover_coords:
            p1_has_wall = parent1.grid[y, x] == CellType.WALL.value
            p2_has_wall = parent2.grid[y, x] == CellType.WALL.value

            # Crossover for offspring1
            o1_has_wall = offspring1.grid[y, x] == CellType.WALL.value
            if o1_has_wall and not p2_has_wall:
                BaseSolver._remove_wall(offspring1, x, y)
            elif not o1_has_wall and p2_has_wall:
                if offspring1.remainingWalls > 0:
                    BaseSolver._add_wall(offspring1, x, y)

            # Crossover for offspring2
            o2_has_wall = offspring2.grid[y, x] == CellType.WALL.value
            if o2_has_wall and not p1_has_wall:
                BaseSolver._remove_wall(offspring2, x, y)
            elif not o2_has_wall and p1_has_wall:
                if offspring2.remainingWalls > 0:
                    BaseSolver._add_wall(offspring2, x, y)

        return offspring1, offspring2

    def _mutate(self, env: PatheryEnv, data_logger: Optional[IO] = None) -> PatheryEnv:
        """
        Performs a random mutation on a chromosome.
        """
        mutated_env = env.copy()
        pre_mutation_state = mutated_env.grid.copy()
        fitness_before = self._calculate_fitness(mutated_env)

        num_walls = np.sum(mutated_env.grid == CellType.WALL.value)

        valid_mutations = []
        if num_walls < mutated_env.wallsToPlace:
            valid_mutations.append("ADD")
        if num_walls > 0:
            valid_mutations.append("REMOVE")
            valid_mutations.append("MOVE")

        if not valid_mutations:
            return mutated_env  # No possible mutation

        mutation_type = random.choice(valid_mutations)

        if mutation_type == "MOVE":
            wall_positions = np.where(mutated_env.grid == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))
            empty_squares = np.where(mutated_env.grid == CellType.OPEN.value)
            empty_squares = list(zip(empty_squares[1], empty_squares[0]))

            if not wall_positions or not empty_squares:
                return mutated_env  # No possible mutation

            wall_to_move = random.choice(wall_positions)
            new_position = random.choice(empty_squares)

            BaseSolver._remove_wall(mutated_env, wall_to_move[0], wall_to_move[1])
            BaseSolver._add_wall(mutated_env, new_position[0], new_position[1])
            mutation_info = {
                "type": "MOVE",
                "from": [int(wall_to_move[0]), int(wall_to_move[1])],
                "to": [int(new_position[0]), int(new_position[1])],
            }

        elif mutation_type == "ADD":
            empty_squares = np.where(mutated_env.grid == CellType.OPEN.value)
            empty_squares = list(zip(empty_squares[1], empty_squares[0]))

            if not empty_squares:
                return mutated_env  # No possible mutation

            new_wall_position = random.choice(empty_squares)
            BaseSolver._add_wall(
                mutated_env, new_wall_position[0], new_wall_position[1]
            )
            mutation_info = {
                "type": "ADD",
                "to": [int(new_wall_position[0]), int(new_wall_position[1])],
            }

        elif mutation_type == "REMOVE":
            wall_positions = np.where(mutated_env.grid == CellType.WALL.value)
            wall_positions = list(zip(wall_positions[1], wall_positions[0]))

            if not wall_positions:
                return mutated_env  # No possible mutation

            wall_to_remove = random.choice(wall_positions)
            BaseSolver._remove_wall(mutated_env, wall_to_remove[0], wall_to_remove[1])
            mutation_info = {
                "type": "REMOVE",
                "from": [int(wall_to_remove[0]), int(wall_to_remove[1])],
            }

        fitness_after = self._calculate_fitness(mutated_env)
        reward = fitness_after - fitness_before

        if data_logger:
            log_entry = {
                "pre_mutation_state": pre_mutation_state.tolist(),
                "mutation_info": mutation_info,
                "reward": reward,
            }
            data_logger.write(json.dumps(log_entry) + "\n")

        return mutated_env

    def _before_selection_hook(self, generation: int):
        """
        A hook that is called before the selection process.
        Subclasses can override this to implement custom logic.
        """
        pass

    def _after_new_population_hook(self, generation: int):
        """
        A hook that is called after a new population is created.
        Subclasses can override this to implement custom logic.
        """
        pass
