# Pathery Puzzle Solver

This project is a high-performance, Python-based command-line application designed to programmatically solve puzzles from the game Pathery. It uses a sophisticated hybrid genetic algorithm, with its most computationally intensive components offloaded to C++, to find optimal or near-optimal solutions for a given puzzle layout.

## Key Features

*   **Hybrid Genetic Algorithm:** The core of the solver is a genetic algorithm that evolves populations of wall configurations to maximize path length.
*   **C++ Accelerated Pathfinding:** The A* pathfinding algorithm, a critical performance bottleneck, is implemented in C++ and seamlessly integrated with Python using `pybind11` for maximum speed.
*   **Parallel Fitness Calculation:** The fitness of each individual in the genetic algorithm's population is calculated in parallel, significantly reducing the time required to evolve solutions.
*   **Extensible Solver Architecture:** The project includes a flexible architecture that allows for the implementation of multiple solver algorithms, including:
    *   Hill Climbing
    *   Simulated Annealing
    *   The Hybrid Genetic Algorithm (the most effective solver)
*   **Comprehensive Tooling:** The project includes a benchmark tool for performance measurement, a suite of unit tests for ensuring correctness, and detailed logging for debugging and analysis.

## Project Overview

*   **`puzzles/`**: This directory contains puzzle definitions in a simple JSON format. Each file defines a puzzle's dimensions, wall count, start/finish points, and rock locations.
*   **`pathery_solver.py`**: The main entry point for the application. It contains the implementation of the genetic algorithm and other solution strategies.
*   **`pathery_emulator.py`**: Simulates the Pathery game board, handling the grid, walls, rocks, and the pathfinding logic.
*   **`pathery_pathfinding.cpp`**: A C++ implementation of the A* pathfinding algorithm, exposed to Python using `pybind11`.
*   **`setup.py`**: The build script responsible for compiling the C++ code into a Python extension.
*   **`test_pathery.py`**: A suite of unit tests built with Python's `unittest` module.
*   **`benchmark.py`**: A script for measuring the performance of the solver.

## Prerequisites

*   Python 3
*   A C++ compiler (e.g., GCC, Clang, MSVC)

## Setup and Installation

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies and Compile the C++ Module:**
    *From within the activated virtual environment*, run the following commands:
    ```bash
    pip install -r requirements.txt
    python setup.py install
    ```

## Usage

**Note:** Before running any of the following commands, make sure you have activated the virtual environment (`source venv/bin/activate`).

### Solving Puzzles

To run the solver on a specific puzzle, provide the path to the puzzle file and the number of generations to run as command-line arguments.

```bash
python3 pathery_solver.py puzzles/puzzle_1.json --generations 50
```

### Running Tests

To run the unit tests, execute the `test_pathery.py` script.

```bash
python3 test_pathery.py
```

### Running Benchmarks

To measure the performance of the solver, run the `benchmark.py` script.

```bash
python3 benchmark.py
```
The script will print the final path length and execution time to the console, and log the result to `benchmark_results.log`.

## Logging

The application generates two log files:
*   `solver.log`: Logs the progress of the genetic algorithm, including the best score at each generation.
*   `benchmark_results.log`: Logs the results of the benchmark runs.
