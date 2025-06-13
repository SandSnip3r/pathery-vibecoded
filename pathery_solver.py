
from multiprocessing import Pool
import math
import random
from pathery_emulator import PatheryEmulator

def _calculate_fitness(args):
    individual, num_walls, width, height, start, finish, rocks = args
    
    # Create a new emulator instance for each process
    emulator = PatheryEmulator(width, height, num_walls)
    emulator.set_start(start[0], start[1])
    emulator.set_finish(finish[0], finish[1])
    for rock in rocks:
        emulator.add_rock(rock[0], rock[1])

    solver = PatherySolver(emulator)
    solver._clear_walls()
    for x, y in individual:
        solver.emulator.add_wall(x, y)
    
    _, path_length, optimized_individual = solver._hill_climb_optimizer(num_walls)
    return path_length, optimized_individual



class PatherySolver:
    """
    A solver for the Pathery puzzle game.
    """

    def __init__(self, emulator):
        """
        Initializes the Pathery solver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
        """
        self.emulator = emulator

    def solve_hill_climbing(self, num_walls, num_restarts):
        """
        Attempts to find the longest path using a hill-climbing algorithm.

        Args:
            num_walls (int): The number of walls to place.
            num_restarts (int): The number of times to restart the algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        best_path = None
        best_path_length = 0
        best_grid = None

        for _ in range(num_restarts):
            self._clear_walls()
            self._randomly_place_walls(num_walls)
            
            _, path_length, _ = self._hill_climb_optimizer(num_walls)

            if path_length > best_path_length:
                best_path_length = path_length
                best_path = self.emulator.find_path()
                best_grid = [row[:] for row in self.emulator.grid]

        # Restore the best grid found
        if best_grid:
            self.emulator.grid = best_grid

        return best_path, best_path_length

    def _hill_climb_optimizer(self, num_walls, max_steps=5, num_samples=5):
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

            # Get empty squares adjacent to the current path
            path_neighbors = set()
            for x_path, y_path in current_path:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x_path + dx, y_path + dy
                    if 0 <= nx < self.emulator.width and 0 <= ny < self.emulator.height and self.emulator.grid[ny][nx] == ' ':
                        path_neighbors.add((nx, ny))
            
            if not path_neighbors:
                break

            for x_wall, y_wall in wall_positions:
                for x_new, y_new in random.sample(list(path_neighbors), min(len(path_neighbors), num_samples)):
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

    def _clear_walls(self):
        for y in range(self.emulator.height):
            for x in range(self.emulator.width):
                if self.emulator.grid[y][x] == '#':
                    self.emulator.remove_wall(x, y)

    def _randomly_place_walls(self, num_walls):
        for _ in range(num_walls):
            while True:
                x = random.randint(0, self.emulator.width - 1)
                y = random.randint(0, self.emulator.height - 1)
                if self.emulator.grid[y][x] == ' ':
                    self.emulator.add_wall(x, y)
                    break

    def solve_simulated_annealing(self, num_walls, initial_temp, cooling_rate):
        """
        Attempts to find the longest path using simulated annealing.

        Args:
            num_walls (int): The number of walls to place.
            initial_temp (float): The initial temperature.
            cooling_rate (float): The rate at which the temperature cools.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self._clear_walls()
        self._randomly_place_walls(num_walls)

        current_path = self.emulator.find_path()
        if not current_path:
            return None, 0

        current_path_length = len(current_path)
        best_path = current_path
        best_path_length = current_path_length
        best_grid = [row[:] for row in self.emulator.grid]

        temp = initial_temp

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
            temp *= 1 - cooling_rate

        # Restore the best grid found
        if best_grid:
            self.emulator.grid = best_grid

        return best_path, best_path_length

    def solve_hybrid_genetic_algorithm(self, num_walls, population_size, num_generations, mutation_rate, elite_size):
        """
        Attempts to find the longest path using a hybrid genetic algorithm.

        Args:
            num_walls (int): The number of walls to place.
            population_size (int): The size of the population in each generation.
            num_generations (int): The number of generations to run.
            mutation_rate (float): The probability of a mutation occurring.
            elite_size (int): The number of top individuals to carry over to the next generation.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        best_path_length = 0
        best_individual = None

        # Initialize population
        population = []
        for _ in range(population_size):
            self._clear_walls()
            self._randomly_place_walls(num_walls)
            wall_positions = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == '#':
                        wall_positions.append((x, y))
            population.append(wall_positions)

        for generation in range(num_generations):
            logging.info(f"Generation {generation + 1}/{num_generations}, Best score so far: {best_path_length}")
            logging.getLogger().handlers[0].flush()
            
            rocks = []
            for y in range(self.emulator.height):
                for x in range(self.emulator.width):
                    if self.emulator.grid[y][x] == 'O':
                        rocks.append((x,y))

            
            # Asynchronously calculate fitness for the population
            with Pool() as pool:
                results = pool.map(
                    _calculate_fitness,
                    [(individual, num_walls, self.emulator.width, self.emulator.height, self.emulator.start, self.emulator.finish, rocks) for individual in population]
                )

            fitness_scores, optimized_individuals = zip(*results)

            for i, score in enumerate(fitness_scores):
                if score > best_path_length:
                    best_path_length = score
                    best_individual = optimized_individuals[i]

            # Select parents and carry over elites
            sorted_population = [x for _, x in sorted(zip(optimized_individuals, optimized_individuals), key=lambda pair: pair[0], reverse=True)]
            elites = sorted_population[:elite_size]
            parents = self._select_parents(optimized_individuals, fitness_scores)

            # Create new population
            new_population = elites
            for _ in range(population_size - elite_size):
                parent1, parent2 = random.choices(parents, k=2)
                child = self._crossover(parent1, parent2, num_walls)
                self._mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population

        # Restore the best grid found
        if best_individual:
            self._clear_walls()
            for x, y in best_individual:
                self.emulator.add_wall(x, y)

        return self.emulator.find_path(), best_path_length

    def _select_parents(self, population, fitness_scores):
        # Simple tournament selection
        parents = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), k=3)
            tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            parents.append(population[winner_index])
        return parents

    def _crossover(self, parent1, parent2, num_walls):
        child = []
        combined = parent1 + parent2
        random.shuffle(combined)
        for gene in combined:
            if gene not in child:
                child.append(gene)
            if len(child) == num_walls:
                break
        return child

    def _mutate(self, individual, mutation_rate):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                while True:
                    new_x = random.randint(0, self.emulator.width - 1)
                    new_y = random.randint(0, self.emulator.height - 1)
                    if self.emulator.grid[new_y][new_x] == ' ' and (new_x, new_y) not in individual:
                        individual[i] = (new_x, new_y)
                        break

import json

def load_puzzle(file_path):
    """
    Loads a puzzle from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        PatheryEmulator: An instance of the PatheryEmulator with the loaded puzzle.
    """
    with open(file_path, 'r') as f:
        puzzle_data = json.load(f)

    game = PatheryEmulator(puzzle_data['width'], puzzle_data['height'], puzzle_data['num_walls'])
    game.set_start(puzzle_data['start'][0], puzzle_data['start'][1])
    game.set_finish(puzzle_data['finish'][0], puzzle_data['finish'][1])

    for rock in puzzle_data['rocks']:
        game.add_rock(rock[0], rock[1])

    return game, puzzle_data['best_solution']

import logging

logging.basicConfig(filename='/usr/local/google/home/victorstone/pathery_project/solver.log', level=logging.INFO, format='%(asctime)s - %(message)s')

import sys

import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle_file", help="The path to the puzzle file.")
    parser.add_argument("--generations", type=int, default=200, help="The number of generations to run.")
    args = parser.parse_args()

    # Load the puzzle
    game, best_known_solution = load_puzzle(args.puzzle_file)

    # Create a solver
    solver = PatherySolver(game)

    # Find the best path using a hybrid genetic algorithm
    best_path, best_path_length = solver.solve_hybrid_genetic_algorithm(game.num_walls, 100, args.generations, 0.01, 5)

    if best_path:
        print(f"Best path found with length: {best_path_length}")
        if best_path_length > best_known_solution:
            print("New best solution found!")
        game.draw_path(best_path)
        game.display()
    else:
        print("No path found.")
