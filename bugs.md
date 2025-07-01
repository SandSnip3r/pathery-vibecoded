# Bug Reports

## Negative Remaining Walls with Genetic Solver

**ID:** 20250701-01
**Status:** Open
**Severity:** Critical

### Description

When running the data collection script (`scripts/run_data_collection.py`) which uses the `genetic` solver, the final board state is invalid. The number of "Remaining walls" is reported as a negative value (`-13`), which should be impossible. This indicates a flaw in how the genetic algorithm is modifying the wall placements on the board, likely adding more walls than are available.

### Steps to Reproduce

1.  Ensure the project is installed in editable mode (`pip install -e .`).
2.  Run the following command from the project root:

    ```bash
    python scripts/run_data_collection.py --puzzles data/puzzles/puzzle_1.json --num_generations 1 --batch_size 1
    ```

### Observed Behavior

The script completes, but the final output displayed on the console shows an invalid board state.

**Incorrect Output:**
```
╭─────��──────────────────────── Solution Found! ───────────────────────────────╮
│ Best path found with length: 28                                              │
│ +-------------------+                                                        │
│ |S| |#|#|#|█|#|#|#|#|                                                        │
│ |#| | | | |█|#|#|#|#|                                                        │
│ |#|#|#|#| |█|#|#|#|#|                                                        │
│ |#|#|#| | |█|#|#|#|#|                                                        │
│ |#| | | |#|█|#|#|#|#|                                                        │
│ |#| |#|#|#|#|#|#|#|#|                                                        │
│ |#| | |#| | | | |#|#|                                                        │
│ |#|#| |#| |#|#| | |#|                                                        │
│ |#|#| | | |#|#|#| | |                                                        │
│ |#|#|#|#|#|#|#|#|#|G|                                                        │
│ +-------------------+                                                        │
│ Remaining walls: -13                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Expected Behavior

The number of remaining walls should always be a non-negative integer. The solver should respect the puzzle's constraints on the total number of walls that can be placed.
