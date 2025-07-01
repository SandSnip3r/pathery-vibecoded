# Pathery Game Rules and Mechanics

## Core Objective

The main goal in Pathery is to create the longest possible path from a starting point to a finishing point on a grid. Puzzles can have multiple starts and multiple goals. This is done by placing a limited number of "walls" to guide a pathfinding AI.

## Pathfinding AI

The pathfinding AI follows a strict and predictable priority system when choosing its direction. This is the most important mechanic to understand to solve the puzzles. The AI's directional preference is:

1.  **Up**
2.  **Right**
3.  **Down**
4.  **Left**

The path will always go as far as possible in the highest priority direction before trying the next one.

## Placing Walls

For each puzzle, you have a limited number of walls to place on the grid. These walls act as barriers, forcing the pathfinding AI to find a different, and hopefully longer, route.

## Special Tiles and Advanced Mechanics

As you progress, the game introduces new elements:

*   **Ice Tiles:** The pathfinding AI will prioritize moving onto ice tiles to find the shortest route. You must strategically place walls to work around this behavior.
*   **Portals:** The path can enter one portal and exit another, adding another layer of complexity to path calculation.
*   **Checkpoints:** Checkpoints are capital letters from A to N. There can exist multiple instances of the same checkpoint. If multiple instances exist, the pathfinding algorithm always finds the shortest path from the current position to the nearest checkpoint, and then repeats, until finally find the shortest path to the nearest goal.

## Scoring

The game uses a "Champion Points" system:

*   **Attempting a map:** 5 points.
*   **Tying the high score:** 10-200 points, depending on how many others have that score.
*   **Winning a map:** A small bonus for having the highest score.
*   **Weekly maps:** All participants get points, with more points for outperforming more players.

## Game Modes

Pathery has several game modes with increasing difficulty:

*   Simple
*   Normal
*   Complex
*   Teleport Madness
*   Ultra Complex
*   Unlimited
