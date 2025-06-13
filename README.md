# Pathery Puzzle Solver

This project is a Python-based command-line application designed to programmatically solve puzzles from the game Pathery. It uses a sophisticated hybrid genetic algorithm to find optimal or near-optimal solutions for a given puzzle layout.

## Features

*   A Python-based emulator for the core Pathery game mechanics.
*   Multiple solver algorithms, including:
    *   Hill Climbing
    *   Simulated Annealing
    *   A Hybrid Genetic Algorithm (the most effective solver)
*   A benchmark tool to measure solver performance.
*   A suite of unit tests to ensure the correctness of the pathfinding logic.
*   Puzzle definitions stored in a simple JSON format.
*   Logging for both solver progress and emulator actions.

## Project Structure

```
pathery_project/
│
├─── puzzles/
│    ├─── puzzle_1.json
│    └─── puzzle_2.json
│
├─── venv/
│
├─── .gitignore
├─── benchmark.py
├─── pathery_emulator.py
├─── pathery_solver.py
├─── README.md
└─── test_pathery.py
```

*   **`pathery_emulator.py`**: Contains the `PatheryEmulator` class, which simulates the game grid, pathfinding logic, and game rules.
*   **`pathery_solver.py`**: Implements the various solving algorithms. This is the main executable for running the solver.
*   **`puzzles/`**: A directory containing puzzle definitions in JSON format.
*   **`test_pathery.py`**: Unit tests for the project, built with Python's `unittest` module.
*   **`benchmark.py`**: A script for measuring the performance of a single generation of the solver.
*   **`rules.md`**: A markdown file detailing the rules of the Pathery game as understood for this project.
*   **`.gitignore`**: Specifies files and directories to be ignored by Git.
*   **`venv/`**: The Python virtual environment directory.

## Setup and Installation

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    **Note:** All subsequent `python` and `pip` commands should be run from within this activated virtual environment.

2.  **Install Dependencies:**
    There are no external dependencies required to run the current version of the project. All necessary modules are part of the Python standard library.

## Usage

### Solving Puzzles

To run the solver on a specific puzzle, provide the path to the puzzle file as a command-line argument.

```bash
python3 pathery_solver.py puzzles/puzzle_1.json
```
The solver will then run and print the best solution it finds to the console.

### Running Tests

To run the unit tests, execute the `test_pathery.py` script.

```bash
python3 test_pathery.py
```

### Running Benchmarks

To measure the performance of a single generation of the solver, run the `benchmark.py` script.

```bash
python3 benchmark.py
```
The script will print the execution time to the console and log the result to `benchmark_results.log`.
