import time
import random
import logging
from typing import List, Tuple, Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from pathery.solvers.base_solver import BaseSolver
import numpy as np
from pathery.rl.dqn_agent import DQNAgent


class HybridGASolver(BaseSolver):
    """
    A solver that uses a hybrid genetic algorithm with a DQN agent.
    """

    def __init__(
        self,
        env: PatheryEnv,
        population_size: int = 100,
        generations: int = 100,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        perf_logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the HybridGASolver.
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
        self.crossover_rate = crossover_rate
        self.population = []
        self.dqn_agent = DQNAgent(env)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a hybrid genetic algorithm.
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
                self.perf_logger.info(
                    f"hybrid_ga,{time.time()},{gen},0,{max_fitness},,,"
                )
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

                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)

                new_population.append(offspring1)
                new_population.append(offspring2)

            self.population = new_population

            # Train the DQN agent
            self.dqn_agent.train_step()

            # Decay epsilon
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

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
        Performs a mutation on a chromosome using the DQN agent.
        """
        pre_mutation_state = chromosome.copy()
        fitness_before = self._calculate_fitness(pre_mutation_state)

        action = self.dqn_agent.choose_action(pre_mutation_state, self.epsilon)

        mutated_chromosome = chromosome.copy()

        if action["type"] == "MOVE":
            from_pos = action["from"]
            to_pos = action["to"]
            mutated_chromosome[from_pos[1], from_pos[0]] = CellType.OPEN.value
            mutated_chromosome[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "ADD":
            to_pos = action["to"]
            mutated_chromosome[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "REMOVE":
            from_pos = action["from"]
            mutated_chromosome[from_pos[1], from_pos[0]] = CellType.OPEN.value

        fitness_after = self._calculate_fitness(mutated_chromosome)
        reward = fitness_after - fitness_before

        self.dqn_agent.replay_buffer.push(pre_mutation_state, action, reward)

        return mutated_chromosome
