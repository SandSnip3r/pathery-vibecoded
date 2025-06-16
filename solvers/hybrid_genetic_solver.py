
import random
import logging
from multiprocessing import Pool
from typing import List, Tuple, Optional
from pathery_emulator import PatheryEmulator
from solvers.base_solver import BaseSolver

def _init_worker(emulator: PatheryEmulator) -> None:
    global solver
    # This is a simplified solver for the worker process
    solver = BaseSolver(emulator)

def _calculate_fitness(individual: List[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:
    
    solver._clear_walls()
    for x, y in individual:
        solver.emulator.add_wall(x, y)
    
    path = solver.emulator.find_path()
    path_length = len(path) if path else 0
    return path_length, individual


class HybridGeneticSolver(BaseSolver):
    """
    A solver that uses a hybrid genetic algorithm.
    """

    def __init__(self, emulator: PatheryEmulator, population_size: int = 100, num_generations: int = 200, mutation_rate: float = 0.01, elite_size: int = 5, best_known_solution: int = 0) -> None:
        """
        Initializes the HybridGeneticSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
            population_size (int): The size of the population in each generation.
            num_generations (int): The number of generations to run.
            mutation_rate (float): The probability of a mutation occurring.
            elite_size (int): The number of top individuals to carry over to the next generation.
            best_known_solution (int): The best known solution length.
        """
        super().__init__(emulator)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_known_solution = best_known_solution

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a hybrid genetic algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        best_path_length = 0
        best_individual = None

        # Initialize population
        population = []
        for _ in range(self.population_size):
            self._clear_walls()
            self._randomly_place_walls(self.emulator.num_walls)
            wall_positions = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == '#':
                        wall_positions.append((x, y))
            population.append(wall_positions)

        with Pool(initializer=_init_worker, initargs=(self.emulator,)) as pool:
            for generation in range(self.num_generations):
                # Dynamic mutation rate
                current_mutation_rate = max(0.01, self.mutation_rate * (0.95 ** generation))

                logging.info(f"Generation {generation + 1}/{self.num_generations}, Best score so far: {best_path_length}, Mutation rate: {current_mutation_rate:.4f}")
                logging.getLogger().handlers[0].flush()
                
                # Asynchronously calculate fitness for the population
                results = pool.map(_calculate_fitness, population)

                fitness_scores, optimized_individuals = zip(*results)

                for i, score in enumerate(fitness_scores):
                    if score > best_path_length:
                        best_path_length = score
                        best_individual = optimized_individuals[i]
                        if self.best_known_solution > 0 and best_path_length >= self.best_known_solution:
                            logging.info(f"Optimal solution found with length: {best_path_length}. Exiting early.")
                            # Restore the best grid found
                            if best_individual:
                                self._clear_walls()
                                for x, y in best_individual:
                                    self.emulator.add_wall(x, y)
                            return best_individual, best_path_length

                # Select parents and carry over elites
                sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
                elites = sorted_population[:self.elite_size]
                parents = self._select_parents(population, fitness_scores)

                # Create new population
                new_population = elites
                for _ in range(self.population_size - self.elite_size):
                    parent1, parent2 = random.choices(parents, k=2)
                    child = self._crossover(parent1, parent2, self.emulator.num_walls)
                    self._mutate(child, current_mutation_rate)
                    new_population.append(child)

                population = new_population

        # Restore the best grid found
        if best_individual:
            self._clear_walls()
            for x, y in best_individual:
                self.emulator.add_wall(x, y)

        return best_individual, best_path_length

    def _select_parents(self, population: List[List[Tuple[int, int]]], fitness_scores: List[int]) -> List[List[Tuple[int, int]]]:
        # Roulette wheel selection
        parents = []
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # If all fitness scores are 0, select randomly
            return random.choices(population, k=len(population))

        for _ in range(len(population)):
            selection_point = random.uniform(0, total_fitness)
            current_sum = 0
            for i, fitness in enumerate(fitness_scores):
                current_sum += fitness
                if current_sum > selection_point:
                    parents.append(population[i])
                    break
        return parents

    def _crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]], num_walls: int) -> List[Tuple[int, int]]:
        child = []
        
        # Uniform crossover
        for i in range(min(len(parent1), len(parent2))):
            if random.random() < 0.5:
                gene = parent1[i]
            else:
                gene = parent2[i]
            
            if gene not in child:
                child.append(gene)

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
                while True:
                    new_x = random.randint(0, self.emulator.width - 1)
                    new_y = random.randint(0, self.emulator.height - 1)
                    if self.emulator.grid[new_y][new_x] == ' ' and (new_x, new_y) not in individual:
                        individual[i] = (new_x, new_y)
                        break
