# Experiment: Reward Shaping for DQN Genetic Solver

**Status: Open**

## Objective

To improve the performance of the `dqn_genetic_solver` by implementing a reward shaping scheme. The goal is to create a more informative reward signal that encourages the agent to explore the solution space more effectively and avoid the overly conservative behavior observed in the previous experiment.

## Hypothesis

By adding small, intermediate rewards and penalties for specific actions (e.g., adding a wall), we can guide the DQN agent towards a more effective policy. A small bonus for a valid wall placement and a larger penalty for blocking the path will provide a clearer learning signal than relying solely on the change in path length.

## Action Plan

### Step 1: Implement Reward Shaping

The reward shaping logic has already been implemented in `src/pathery/solvers/dqn_genetic_solver.py`. The new reward function is as follows:

-   `reward = fitness_after - fitness_before`
-   If the mutation is `ADD`:
    -   `reward += 0.1` if the new wall does not block the path.
    -   `reward -= 0.2` if the new wall blocks the path.

### Step 2: Data Collection

A new dataset will be collected using the `dqn_genetic_solver` with the reward shaping logic.

**Action:**
1.  Run `scripts/run_data_collection.py` with the `dqn_genetic` solver.
2.  Store the new data in a separate directory, e.g., `output/dqn_ga_transitions_reward_shaping`.

### Step 3: Preprocessing

The new data will be preprocessed into `.pkl` files for training.

**Action:**
1.  Run `scripts/preprocess_logs.py` on the new data directory.
2.  The output will be stored in a corresponding `data/preprocessed_dqn_ga_transitions_reward_shaping` directory.

### Step 4: Fine-Tuning

The existing pre-trained model will be fine-tuned on the new dataset.

**Action:**
1.  Run `scripts/train_dqn.py`, loading the original model from `output/checkpoints`.
2.  The fine-tuned model will be saved to a new directory, e.g., `output/checkpoints_reward_shaping`.

### Step 5: Evaluation

The performance of the fine-tuned model will be evaluated against the baseline `genetic` solver.

**Action:**
1.  Update `config.json` to point the `dqn_genetic` solver to the new fine-tuned model.
2.  Run `scripts/benchmark.py` to generate a new performance report.
3.  Compare the results to the baseline `genetic` solver and the previous `dqn_genetic` solver.

## Expected Outcome

The fine-tuned model with reward shaping will demonstrate a significant improvement in performance over the baseline `genetic` solver and the previous iteration of the `dqn_genetic` solver.
