# Pathery Puzzle Solver

> **Important Note for Developers:** All Python and pip commands for this project **must** be executed using the executables in the `venv/bin` directory (e.g., `venv/bin/python` and `venv/bin/pip`). This ensures that you are using the correct dependencies and that your changes do not affect other Python projects on your system.

This project is a high-performance, Python-based command-line application designed to programmatically solve puzzles from the game Pathery. It uses a variety of sophisticated solver algorithms, with its most computationally intensive components offloaded to C++, to find optimal or near-optimal solutions for a given puzzle layout.

## Key Features

*   **Multiple Solver Algorithms:** The project includes a flexible architecture that allows for the implementation of multiple solver algorithms, including:
    *   **Hill Climbing:** A simple and fast optimization algorithm.
    *   **Simulated Annealing:** A probabilistic technique for approximating the global optimum.
    *   **Hybrid Genetic Algorithm:** A sophisticated algorithm that evolves populations of wall configurations to maximize path length.
    *   **Memetic Algorithm:** A hybrid of a genetic algorithm and a local search algorithm (Hill Climbing).
*   **C++ Accelerated Pathfinding:** The A* pathfinding algorithm, a critical performance bottleneck, is implemented in C++ and seamlessly integrated with Python using `pybind11` for maximum speed.
*   **Parallel Fitness Calculation:** The fitness of each individual in the genetic algorithm's population is calculated in parallel, significantly reducing the time required to evolve solutions.
*   **Extensible Solver Architecture:** The project includes a flexible architecture that allows for the easy addition of new solver algorithms.
*   **Comprehensive Tooling:** The project includes a benchmark tool for performance measurement, a suite of unit tests for ensuring correctness, and detailed logging for debugging and analysis.

## Project Structure

*   **`puzzles/`**: This directory contains puzzle definitions in a simple JSON format. Each file defines a puzzle's dimensions, wall count, start/finish points, rock locations, and the best-known solution length (if available). A value of `0` for `best_solution` indicates that the optimal solution is not known.
*   **`solvers/`**: This directory contains the different solver implementations.
*   **`experiments/`**: This directory contains markdown files that document experiments with different solver algorithms and parameters. It also includes a `template.md` for creating new experiment files.
*   **`pathery_solver.py`**: The main entry point for the application. It loads a puzzle and a solver and runs the solver.
*   **`pathery_emulator.py`**: Simulates the Pathery game board, handling the grid, walls, rocks, and the pathfinding logic.
*   **`pathery_pathfinding.cpp`**: A C++ implementation of the A* pathfinding algorithm, exposed to Python using `pybind11`.
*   **`pathery_rules.md`**: A markdown file containing the rules of the Pathery game.
*   **`setup.py`**: The build script responsible for compiling the C++ code into a Python extension.
*   **`test_pathery.py`**: A suite of unit tests built with Python's `unittest` module.
*   **`benchmark.py`**: A script for measuring the performance of the solver.
*   **`config.json`**: A configuration file for the project, including paths to puzzle files, log files, and solver parameters.

## Prerequisites

*   Python 3
*   A C++ compiler (e.g., GCC, Clang, MSVC)

## Setup and Installation

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

2.  **Install Dependencies and Compile the C++ Module:**
    ```bash
    venv/bin/pip install -r requirements.txt
    venv/bin/python setup.py install
    ```

## Usage

### Solving Puzzles

To run the solver on a specific puzzle, provide the name of the puzzle file and optionally specify a solver.

```bash
venv/bin/python pathery_solver.py puzzle_1 --solver memetic
```

The following solvers are available:
*   `hill_climbing`
*   `simulated_annealing`
*   `hybrid_genetic`
*   `memetic` (default)

### Running Tests

To run the unit tests, execute the `test_pathery.py` script.

```bash
venv/bin/python test_pathery.py
```

### Running Benchmarks

To measure the performance of the solver, run the `benchmark.py` script.

```bash
venv/bin/python benchmark.py
```
The script will print the final path length and execution time to the console, and log the result to `benchmarks/solver_performance.md`.

## Logging

The application generates three log files:
*   `solver.log`: Logs the progress of the genetic algorithm, including the best score at each generation.
*   `benchmark_results.log`: Logs the results of the benchmark runs.
*   `test.log`: Logs the results of the test runs.
