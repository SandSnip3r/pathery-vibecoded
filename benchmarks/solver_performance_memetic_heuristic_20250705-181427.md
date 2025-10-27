# Solver Performance Report

## Benchmark Details

- **Solver(s):** memetic_heuristic
- **Timestamp:** 20250705-181427
- **Runs per Puzzle:** 5

### Configuration for `memetic_heuristic`

```json
{
  "population_size": 100,
  "generations": 100,
  "tournament_size": 3,
  "mutation_rate": 0.8,
  "crossover_rate": 0.8,
  "elitism_size": 2,
  "q_value_threshold": 0.5,
  "local_search_generations": 10,
  "local_search_iterations": 10,
  "model_path": "output/checkpoints_backup_20250704-223429/"
}
```


## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_10.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 729 | 872 | 797.60 | 79.2052 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_20.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 752 | 1109 | 938.40 | 116.0098 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_30.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 758 | 1051 | 915.60 | 87.7631 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_40.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 576 | 1002 | 803.40 | 70.7490 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_50.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 711 | 968 | 831.40 | 69.2628 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_60.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 847 | 1062 | 920.80 | 111.7850 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_70.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 822 | 956 | 890.40 | 91.0476 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_80.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 590 | 834 | 741.20 | 75.0241 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_90.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 649 | 936 | 751.80 | 81.3466 |

## Puzzle: data/puzzles/data/puzzles/ucu/puzzle_100.json

| Solver | Min Path | Max Path | Mean Path | Mean Duration (s) |
|---|---|---|---|---|
| memetic_heuristic | 713 | 940 | 796.20 | 83.2278 |
