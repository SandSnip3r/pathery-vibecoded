# Memetic Solver Improvement Plan

The current `MemeticSolver` uses a fixed 80/20 time split between the genetic algorithm and the hill-climbing optimizer. This is suboptimal. A better approach is to interleave the two algorithms for a more adaptive and robust solver.

## Key Implementation Steps:

1.  **Refactor for Inheritance:**
    *   Create a `BaseGeneticSolver` class that contains the common genetic algorithm logic (population initialization, selection, crossover, mutation).
    *   Have both `HybridGeneticSolver` and `MemeticSolver` inherit from this base class to reduce code duplication and improve maintainability.

2.  **Interleave Algorithms in `MemeticSolver`:**
    *   Modify the main loop in `MemeticSolver` to run for a set number of generations.
    *   Instead of a single, separate hill-climbing phase, apply the hill climber periodically to the best-performing individuals (the "elites") within the main generation loop (e.g., every 10 generations).

3.  **Dynamic Time Allocation:**
    *   The time limit for each local search (hill climbing) should be calculated dynamically based on the remaining time and the number of individuals being improved. This ensures the solver respects the overall time limit without needing a fixed split.

4.  **Integrate Improved Individuals:**
    *   The individuals improved by the hill climber should be directly updated or injected back into the population. This raises the overall fitness and guides the genetic search toward more promising areas of the solution space.

This approach creates a synergistic loop where the genetic algorithm explores the solution space broadly, and the hill climber intensively exploits the most promising areas found, leading to better solutions without manual tuning.
