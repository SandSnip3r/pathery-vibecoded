# DQN-Guided Genetic Algorithm Experiment

This document details an experiment comparing the performance of a standard genetic algorithm with a modified version that uses a pre-trained Deep Q-Network (DQN) to guide its mutation operations.

## Hypothesis

The hypothesis is that a genetic algorithm using a DQN to guide mutations will find better solutions (longer paths) than a standard genetic algorithm with random mutations, given the same number of generations.

## Experimental Setup

- **Puzzle:** `data/puzzles/ucu/puzzle_7.json`
- **Solvers:**
    - `genetic`: A standard genetic algorithm with random mutation.
    - `dqn_genetic`: A genetic algorithm using a pre-trained DQN to select mutations.
- **Parameters:**
    - The experiment was run with several configurations, varying the number of generations and runs. The key parameters for the genetic algorithm are:
        - **Population Size:** 100
        - **Tournament Size:** 3
        - **Mutation Rate:** 0.1
        - **Crossover Rate:** 0.8
- **Runs:** The number of runs for each experiment is specified in the results tables.

## Results

### Run 1: 20 Generations (1 run)

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| genetic | 524 | 524 | 524.00 | 61.1164 |
| dqn_genetic | 506 | 506 | 506.00 | 64.3779 |

### Run 2: 40 Generations (5 runs)

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| genetic | 479 | 637 | 534.40 | 63.3277 |
| dqn_genetic | 488 | 556 | 528.80 | 71.3823 |

### Run 3: 500 Generations (1 run)

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| genetic | 677 | 677 | 677.00 | 114.6238 |
| dqn_genetic | 551 | 551 | 551.00 | 183.4693 |

## Conclusion

The experiments, conducted with an improved population initialization strategy, show that the standard `genetic` solver generally outperforms the `dqn_genetic` solver, especially in longer runs. While the `dqn_genetic` solver is competitive in shorter runs, the `genetic` solver's random mutation strategy appears to be more effective at exploring the solution space and finding better solutions in the long run.

The improved initialization, which creates a more diverse set of starting solutions, seems to have a greater positive impact on the standard `genetic` solver. The random nature of its mutations, combined with a diverse population, allows it to effectively explore the solution space.

Conversely, the `dqn_genetic` solver, even with a diverse starting population, appears to be converging prematurely to suboptimal solutions. This suggests that the DQN model, in its current state, may be too "greedy," favoring short-term gains and failing to explore the solution space as effectively as the random-mutation approach.

These results indicate that for this problem, a purely random mutation strategy, when coupled with a diverse initial population, is more effective than the current DQN-guided approach. Future work could explore hybrid models that incorporate both DQN-guided mutations and a degree of randomness to strike a better balance between exploration and exploitation.
