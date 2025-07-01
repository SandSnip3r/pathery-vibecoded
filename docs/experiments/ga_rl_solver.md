### **Project Plan: A Hybrid GA-RL Solver for Pathery**

**Last Updated:** June 30, 2025, 11:21 AM PDT


### **Part 1: Algorithm Overview**

This solver intelligently combines a Genetic Algorithm (GA) with a Deep Reinforcement Learning (RL) agent. The GA explores the broad solution space, while the RL agent learns to perform sophisticated, high-value mutations, guiding the search toward optimal solutions.


#### **1.1. The Genetic Algorithm (GA) Core**

The GA manages the high-level evolution of a population of potential solutions.



* **Chromosome:** A 27x19 grid representing a single solution. Each cell contains an identifier for the object at that location (e.g., EMPTY, WALL, ICE, TELEPORTER_IN, etc.).
* **Fitness Function:** A function that takes a chromosome as input, runs the Pathery pathfinding logic (including ice slides, teleporters, etc.), and returns a single float value: the length of the resulting path. An invalid path (start disconnected from goal) should return a fitness of 0.
* **Selection Operator:** **Tournament Selection**. To select a parent, a random group of k individuals (start with k=3) is chosen from the population. The one with the highest fitness wins and becomes a parent.
* **Crossover Operator:** **Two-Point Crossover**. Two random (x, y) coordinates on the grid define a rectangle. For two parent chromosomes, the wall placements within this rectangle are swapped to create two offspring.


#### **1.2. The RL Agent (The "Mutation Brain")**

The RL Agent is a Deep Q-Network (DQN) that replaces the GA's standard random mutation. Its sole job is to analyze a single chromosome and decide on the most beneficial mutation to perform.



* **Framework:** This is a **Contextual Bandit** problem solved using DQN architecture. The agent learns to make a single, optimal decision for a given context (the board state). The episode length is 1.
* **Network Architecture:** A Convolutional Neural Network (CNN) with three separate output "heads".
    * **Input:** A 27 x 19 x N tensor representing the board state, where N is the number of channels in a one-hot encoding of the tile types.
    * **Body:** A shared series of convolutional layers that learn to extract spatial features from the board.
    * **Output Head 1 (Removal Scores):** A 27x19 linear output layer that produces a "heatmap" of Q-values for removing a wall at each position.
    * **Output Head 2 (Placement Scores):** A 27x19 linear output layer that produces a "heatmap" of Q-values for placing a wall at each position.
    * **Output Head 3 (Action Type Scores):** A small linear output layer with 3 nodes, producing Q-values for the actions: [MOVE, ADD, REMOVE].


### **Part 2: Implementation Steps**

Follow these steps sequentially to build the solver.

Step 1: Implement the Baseline Genetic Algorithm

Create a new solver class. Implement the core GA logic using the operators defined in Part 1. For this baseline version, the mutation operator should be simple and random (e.g., with a small probability, randomly move one wall to a new valid empty location). This version must be fully functional and able to produce solutions.

Step 2: Implement the Data Logger

Modify the baseline GA's mutation step. Every time a random mutation is performed, log the following data to a file as a single record:



1. pre_mutation_state: The full 27x19 board state *before* the mutation.
2. mutation_info: An object detailing the action taken (e.g., {type: 'MOVE', from: [x1, y1], to: [x2, y2]}).
3. reward: The change in fitness caused by the mutation (fitness_after - fitness_before).

Step 3: Build the RL Agent (DQN)

Create a new class for your DQN agent.



1. **Network:** Implement the three-headed CNN architecture described in section 1.2 using JAX.
2. **Experience Replay Buffer:** Implement a class to store the experiences collected by the logger. It should have a method to store(state, action, reward) and a method to sample(batch_size) that returns a random minibatch of experiences.
3. **Loss Function:** The loss will be the Mean Squared Error (MSE) between the network's predicted Q-value for the action taken and the target value. Since the episode length is 1, the target Q-value is simply the observed reward, y = r.
4. **Training Method:** Create a train_step method that samples a minibatch from the replay buffer, performs a forward pass, calculates the loss, and performs backpropagation to update the network weights.

Step 4: Integrate the RL Agent into the GA

Create the final hybrid solver class.



1. Instantiate the GA and the (untrained) DQN agent.
2. Replace the GA's call to the random mutation operator with a call to the DQN agent.
3. Implement the agent's action-selection logic (choose_action method):
    * It takes the chromosome (board state) and an epsilon value as input.
    * With probability epsilon, it performs a random mutation (to enable exploration).
    * With probability 1-epsilon, it performs an "intelligent" mutation:
        * Passes the state through the DQN to get the three output heads.
        * Determines the best action type from the Action Type Head.
        * Based on the action type, finds the best move(s) from the Removal and/or Placement heads.
        * Applies this mutation to the chromosome.
4. Ensure that every mutation performed (both random and intelligent) is logged to the experience replay buffer.


### **Part 3: Training Protocol**

This is a three-phase process.

**Phase 1: Offline Data Generation**



* **Goal:** Populate the Experience Replay Buffer with a large, diverse dataset.
* **Action:** Run the **Baseline GA solver** (from Step 1) on a wide variety of Pathery puzzles. Let it run for an extended period. Aim to collect **100,000 to 500,000** experience tuples. Save this dataset to disk.

**Phase 2: Offline Q-Network Training**



* **Goal:** Pre-train the DQN agent on the static dataset to give it a strong initial policy.
* **Action:**
    1. Load the dataset from Phase 1 into the DQN's Experience Replay Buffer.
    2. Set your hyperparameters:
        * **Minibatch Size:** 128 or 256.
        * **Learning Rate:** Start with 1e-4.
        * **Epochs:** 5 to 20.
    3. Run a training loop. For each epoch, iterate through the entire dataset in minibatches, calling the agent's train_step method for each batch.
    4. Save the trained model weights.

**Phase 3: Online Fine-Tuning and Solving**



* **Goal:** Use the pre-trained agent to solve the target puzzle, allowing it to continue learning as it goes.
* **Action:**
    1. Run the **final Hybrid GA-RL solver** (from Step 4).
    2. Load the pre-trained model weights into your DQN agent.
    3. Set the epsilon-annealing schedule. Start epsilon at 1.0.
    4. Over the course of the GA run (e.g., over 10,000 generations), linearly decay epsilon from 1.0 down to a final value of 0.05.
    5. During the run, the agent will select actions and the GA will update its population. After each generation (or every N generations), call the agent's train_step method one or more times to fine-tune the network on the newest experiences it has gathered.
