# Bug: `hybrid_ga` solver is broken

The `hybrid_ga` solver is currently broken due to an `AttributeError`.

## Reproducing the bug

Run the following command:

```bash
python3 src/pathery/main.py data/puzzles/puzzle_1.json --solver hybrid_ga
```

## Error

The solver will fail with the following error:

```
AttributeError: 'DQNAgent' object has no attribute 'train_step'
```

## Details

The `hybrid_ga_solver.py` script is calling `self.dqn_agent.train_step()`, but the `DQNAgent` class does not have a `train_step` method.
