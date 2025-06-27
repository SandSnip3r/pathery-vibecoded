# Emulator Migration

This document details the migration from the old `pathery_emulator.py` to the new `PatheryEnv` gymnasium environment.

## Changes Made

The following changes were made during the migration:

- The `pathery_emulator.py` file was deleted.
- The `pathery_env_adapter.py` file was removed and all dependent files were updated to use the `PatheryEnv` directly.
- A `MapBuilder` utility was created to simplify the creation of test puzzles.
- The tests were updated to use the new `MapBuilder` and `PatheryEnv`.
- The solvers were updated to use the new `PatheryEnv`.
- The `pathery_solver.py` file was updated to use the new `PatheryEnv`.
- The `PatheryEnv` was modified to support setting the start, finish, and checkpoint positions.
- The `PatheryEnv` was modified to accept a `map_string` parameter in its constructor. This allows for the creation of a `PatheryEnv` with a specific width, height, and number of walls.
- The C++ pathfinding library for `PatheryEnv` was compiled to improve performance.
- The `_randomly_place_walls` method in `base_solver.py` was made more efficient.
- The `HybridGeneticSolver` was improved to be more robust.

## Justifications

The migration was performed to take advantage of the more robust and feature-rich `PatheryEnv` gymnasium environment. The `PatheryEnv` provides a number of advantages over the old emulator, including:

- A more standardized interface.
- Support for a wider range of puzzle features, such as ice and teleporters.
- A more efficient and robust pathfinding implementation.

The changes made during the migration were necessary to ensure that the project continues to function correctly after the switch to the new environment.
