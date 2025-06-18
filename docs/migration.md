# Emulator Migration

This document details the migration from the old `pathery_emulator.py` to the new `PatheryEnv` gymnasium environment.

## Changes Made

The following changes were made during the migration:

- The `pathery_emulator.py` file was deleted.
- A new `pathery_env_adapter.py` file was created to act as a compatibility layer between the old emulator's interface and the new `PatheryEnv`'s interface.
- All dependent files were updated to use the new `pathery_env_adapter.py` instead of the old emulator.
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

## Next Steps

The following next steps should be taken to clean up after the migration:

- The `pathery_env_adapter.py` file should be removed and the dependent files should be updated to use the `PatheryEnv` directly. This will require a significant amount of refactoring, but it will result in a cleaner and more maintainable codebase.
- The `PatheryEnv` should be modified to support all the features of the old emulator, such as drawing the path on the grid.
- The solvers should be updated to take advantage of the new features in the `PatheryEnv`, such as ice and teleporters.

## Information for New Users

New users of the project will need to be aware of the following:

- The `pathery_emulator.py` file no longer exists. All interaction with the emulator should be done through the `PatheryEnv` gymnasium environment.
- The `PatheryEnv` requires the `gymnasium` and `numpy` python packages to be installed.
- The C++ pathfinding library for `PatheryEnv` must be compiled before running the tests or solvers. This can be done by running the `make` command in the `PatheryEnv/pathery_env/cpp_lib/` directory.
- The `PYTHONPATH` environment variable must be set to include the `PatheryEnv` directory. This can be done by running the following command:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/pathery_project/PatheryEnv
```
