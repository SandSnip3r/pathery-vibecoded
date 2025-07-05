# Experiment: Feasibility of Online Reinforcement Learning

**Status: Closed**

## Objective

To analyze the computational feasibility of implementing a true online reinforcement learning approach for the `dqn_genetic` solver, where the DQN model is trained continuously within the solver's execution loop.

## Background

Previous experiments with offline-trained RL models have failed to outperform the baseline `genetic` solver. The most successful strategy discovered so far is a `genetic` solver with a very high, purely random mutation rate.

A key hypothesis for the failure of the offline models is **distribution shift**: the models were trained on a static dataset that does not accurately represent the data distribution encountered during the actual solving process. A true online RL approach, where the agent learns from its own actions in a continuous feedback loop, would solve this problem.

This document assesses whether such an approach is computationally practical.

## Compute Time Analysis

The analysis is based on a single run of the solver with the following configuration: `generations: 100`, `population_size: 100`.

| Component | Description | Time per Call (est.) | Calls per Run | Total Time per Run |
| :--- | :--- | :--- | :--- | :--- |
| **Fitness Calculation (A\*)** | CPU-bound C++ pathfinding | ~10 ms | ~9,900 | ~99 seconds |
| **DQN Inference** | GPU forward pass (model chooses action) | ~3 ms | ~9,800 | ~29 seconds |
| **Online Training Step** | GPU forward & backward pass | ~31 ms | ~9,800 | ~304 seconds |

### Scenario A: Current Offline Solver

The current solver only performs fitness calculation and inference.

-   **Total Time:** ~99s (CPU) + ~29s (GPU) = **~128 seconds**
-   **Primary Bottleneck:** CPU (A\* pathfinding) at ~77% of compute time.
-   *This estimate aligns with our observed benchmark times of ~2 minutes per run.*

### Scenario B: Proposed Online Solver

This scenario adds a training step after every mutation.

-   **Total Time:** ~128s (from Scenario A) + ~304s (GPU Training) = **~432 seconds**
-   **Primary Bottleneck:** GPU (Online Training) at ~70% of compute time.

## Conclusion

Implementing a true online RL training loop would increase the solver's run time from **~2 minutes to over 7 minutes**, a **~3.5x slowdown**.

While online training is the theoretically correct way to apply Deep Q-Learning, the practical computational cost makes it an unviable strategy for this problem. The primary bottleneck would shift from the CPU to the GPU, and the long feedback loop would make any further experimentation, such as hyperparameter tuning, prohibitively slow.

Therefore, the recommendation is to **not pursue online RL at this time**. The most effective path forward remains the optimized `genetic` solver that leverages a high, random mutation rate. This experiment is now closed.
