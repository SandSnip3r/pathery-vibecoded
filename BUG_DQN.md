# Bug: `dqn_genetic` solver is broken

The `dqn_genetic` solver is currently broken due to a `ValueError` when loading the model checkpoint.

## Reproducing the bug

Run the following command:

```bash
python3 src/pathery/main.py data/puzzles/puzzle_1.json --solver dqn_genetic
```

## Error

The solver will fail with the following error:

```
ValueError: Requested shape: (6400, 513) is not compatible with the stored shape: (32832, 513). Truncating/padding is disabled.
```

## Details

The error occurs because the model is trying to load a checkpoint with a different shape than the current model. This is likely because the model was trained on a different set of puzzles with different dimensions, and the current model is being initialized with the dimensions of `puzzle_1.json`.
