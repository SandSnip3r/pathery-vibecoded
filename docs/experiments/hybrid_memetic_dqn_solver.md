# Experiment: Memetic Solver with DQN-Powered Heuristic

## Hypothesis

The purely random genetic algorithm is highly effective at exploration but lacks a refinement mechanism. A memetic algorithm (GA + local search) can improve upon this by refining promising solutions.

This experiment proposes using our existing, pre-trained DQN not as a decision-making agent, but as a simple, computationally cheap **heuristic function**. Its purpose is to decide *when* to trigger a local search, without interfering with the GA's proven random exploration strategy.

## Redefining the DQN's Role

To be clear, this approach abandons the concept of the DQN as an active, decision-making agent. Instead, we are re-purposing the trained network to serve a much simpler function.

The DQN's action space is effectively collapsed from thousands of possible moves into a single binary decision: **"Refine" or "Don't Refine"**.

## Proposed Algorithm: `memetic_heuristic`

This new solver will operate as follows:

1.  **Primary Solver:** A standard `genetic` solver will run with the optimal high-exploration hyperparameters (`mutation_rate=0.8`, `epsilon=1.0`).
2.  **Heuristic Trigger:** After a set number of generations (e.g., every 10), the solver will feed the current best solution into the pre-trained DQN.
3.  **The "Refine/Don't Refine" Decision:** The solver will examine the Q-values produced by the DQN's output heads. If the maximum Q-value across all possible actions exceeds a predefined threshold, the DQN is effectively signaling that it sees a high-potential move. This will be interpreted as a "Refine" decision. Otherwise, it's a "Don't Refine" decision.
4.  **Local Search Activation:** If the decision is "Refine," a hill-climbing local search is initiated on the promising solution. If the decision is "Don't Refine," the GA continues its normal operation without interruption.
5.  **Hill-Climbing:** The local search algorithm will iteratively try small modifications to find an improved version of the solution.
6.  **Population Update:** If the local search finds a better solution, it replaces the original in the population.

This approach leverages each component for its strengths:
*   The **GA** provides broad, random exploration.
*   The **DQN** is used as a fast, learned heuristic to identify candidates for refinement, without dictating any specific moves.
*   The **Hill-Climber** performs the actual, targeted refinement.

## Experimental Plan

### Step 1: Implement the `memetic_heuristic` Solver

A new solver class will be created. It will integrate the `genetic` solver, the `DQNAgent` (as a heuristic), and a hill-climbing function.

### Step 2: Establish a Control Group

The current best solver (the highly-random `genetic` solver) will be run on a benchmark suite of 10-15 puzzles to establish a clear performance baseline.

### Step 3: Run the Experiment

The new `memetic_heuristic` solver will be run on the same benchmark suite. Key parameters to tune will be:
*   The frequency of the heuristic trigger (e.g., every 5, 10, or 20 generations).
*   The Q-value threshold for making the "Refine" decision.
*   The number of iterations for the hill-climbing algorithm.

### Step 4: Analyze the Results

The performance of the `memetic_heuristic` solver will be compared against the control group based on:
*   Average path length.
*   Time to find the best solution.
*   The frequency with which the local search is triggered and whether it leads to improvements.

## Expected Outcome

It is expected that the `memetic_heuristic` solver will find solutions that are, on average, better than the purely random `genetic` solver. By using the DQN as a targeted trigger for local search, the algorithm should be able to refine promising solutions without sacrificing the broad exploration that has proven to be so effective. This should lead to a new best-performing solver.
