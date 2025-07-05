# Pathery Game Rules and Mechanics

## Core Objective

The main goal in Pathery is to create the longest possible path from a starting point to a finishing point on a grid. This is done by placing a limited number of "walls" to guide a pathfinding AI.

Puzzles can have multiple start and goal positions. In cases with multiple starts, the pathfinding AI will calculate the shortest path from all available starting positions and select the one that results in the shortest overall path.

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

*   **Ice Tiles:** The path cannot turn on an ice tile. If the path enters an ice tile, it must continue in a straight line until it is no longer on an ice tile.
*   **Teleporters:** The pathfinding algorithm ignores teleporters when calculating the shortest route. A path might pass through a teleporter, but the teleporter itself does not influence the initial path calculation. If a path does enter a teleporter, it will emerge from a corresponding exit point, and the path will then be recalculated to the original destination.
*   **Checkpoints:** Checkpoints are capital letters from A to N. The path must pass through all checkpoints in alphabetical order before heading to the goal. If multiple instances of the same checkpoint exist, the pathfinding algorithm will choose the one that results in the shortest path to that checkpoint.

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
