# DQN Model Architecture for Pathery

This document outlines a proposed neural network architecture for the Pathery DQN agent, designed to improve performance and better capture the game's complex mechanics.

## 1. Understanding the Problem

Pathery is a puzzle game where the goal is to guide a pathfinding AI by placing a limited number of walls. The game's complexity comes from the interaction of the AI's predictable pathfinding logic with various special tiles like ice and portals.

### 1.1. Observation Space

The input to our model is a one-hot encoded representation of the game board. The board is 27x19, and each cell can have one of 34 possible types (e.g., open, wall, start, goal, ice, portal, etc.). This results in an input shape of `(27, 19, 34)`.

### 1.2. Action Space

The agent can perform three types of actions:

-   **ADD**: Place a wall on an empty cell.
-   **REMOVE**: Remove an existing wall.
-   **MOVE**: Move a wall from one cell to another.

The action space is complex, as it involves not only choosing an action type but also the position(s) for that action.

## 2. Current Model Architecture

The current model is a simple Convolutional Neural Network (CNN) with three output heads:

-   `removal_scores`: A dense layer with `27 * 19` outputs, representing the score for removing a wall at each position.
-   `placement_scores`: A dense layer with `27 * 19` outputs, representing the score for placing a wall at each position.
-   `action_type_scores`: A dense layer with 3 outputs, representing the scores for the `ADD`, `REMOVE`, and `MOVE` actions.

### 2.1. Limitations of the Current Model

While this model is a good starting point, it has some limitations:

-   **Limited Receptive Field**: The three convolutional layers with 3x3 kernels have a limited receptive field, which may not be sufficient to capture long-range dependencies and complex pathing patterns.
-   **No Explicit Path Information**: The model does not explicitly consider the path that results from an action, which is the primary factor determining the reward.
-   **Simple Action Combination**: The final Q-value is a simple sum of the action type and position scores, which may not be expressive enough to capture the complex interactions between them.

## 3. Proposed Model Architecture: A Residual Network with Attention

To address these limitations, I propose a new architecture based on a Residual Network (ResNet) with an attention mechanism.

```
Input (27, 19, 34)
│
├─> Conv(64, (3, 3), padding='same')
│
├─> ResidualBlock(64) x 4
│
├─> AttentionBlock(64)
│
├─> Flatten
│
├─> Dense(512)
│
├─┬─> removal_scores (Dense(27 * 19))
│ ├─> placement_scores (Dense(27 * 19))
│ └─> action_type_scores (Dense(3))
```

### 3.1. Key Components

-   **Residual Blocks**: The use of residual blocks will allow us to build a deeper network without suffering from the vanishing gradient problem. This will increase the model's receptive field and allow it to learn more complex features.
-   **Attention Mechanism**: An attention block will allow the model to focus on the most relevant parts of the board when making a decision. This is particularly important for puzzles with special tiles like portals, where the model needs to understand the long-range dependencies between them.
-   **Shared Feature Extractor**: The ResNet and attention blocks will act as a shared feature extractor, learning a rich representation of the board state that can be used by all three output heads.

### 3.2. Rationale

This architecture is a better fit for the Pathery problem for the following reasons:

-   **Improved Feature Extraction**: The deeper ResNet will be able to learn more complex and abstract features, which is crucial for understanding the game's mechanics.
-   **Focus on Relevant Information**: The attention mechanism will allow the model to focus on the most important parts of the board, such as the start and end points, special tiles, and the current path.
-   **More Expressive Q-Values**: The shared feature extractor will provide a richer set of features to the output heads, allowing for more expressive and accurate Q-value calculations.

## 4. Conclusion

The proposed ResNet architecture with an attention mechanism offers a significant improvement over the current model. It is better suited to handle the complexity of the Pathery game and has the potential to achieve much better performance.
