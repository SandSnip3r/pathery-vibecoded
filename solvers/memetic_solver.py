
from solvers.base_solver import BaseSolver
from solvers.hybrid_genetic_solver import HybridGeneticSolver
from solvers.hill_climbing_solver import HillClimbingSolver

class MemeticSolver(BaseSolver):
    """
    A solver that uses a memetic algorithm.
    """

    def __init__(self, emulator, population_size=100, num_generations=200, mutation_rate=0.01, elite_size=5, best_known_solution=0, **kwargs):
        """
        Initializes the MemeticSolver.

        Args:
            emulator (PatheryEmulator): An instance of the PatheryEmulator.
            population_size (int): The size of the population in each generation.
            num_generations (int): The number of generations to run.
            mutation_rate (float): The probability of a mutation occurring.
            elite_size (int): The number of top individuals to carry over to the next generation.
            best_known_solution (int): The best known solution length.
        """
        super().__init__(emulator, best_known_solution)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.hill_climbing_restarts = kwargs.get("hill_climbing_restarts", 5)

    def solve(self):
        """
        Attempts to find the longest path using a memetic algorithm.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        # First, run the genetic algorithm to find a good starting solution
        genetic_solver = HybridGeneticSolver(
            self.emulator,
            self.population_size,
            self.num_generations,
            self.mutation_rate,
            self.elite_size,
            self.best_known_solution
        )
        best_individual, _ = genetic_solver.solve()

        # If the genetic algorithm didn't find a solution, return
        if not best_individual:
            return None, 0

        # Now, refine the best solution using hill climbing
        self._clear_walls()
        for x, y in best_individual:
            self.emulator.add_wall(x, y)

        hill_climbing_solver = HillClimbingSolver(self.emulator, num_restarts=self.hill_climbing_restarts)
        best_path, best_path_length, _ = hill_climbing_solver._hill_climb_optimizer(self.emulator.num_walls)

        return best_path, best_path_length
