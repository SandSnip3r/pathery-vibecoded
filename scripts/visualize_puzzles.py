import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.utils import load_puzzle


def visualize_puzzle(puzzle_name: str):
    """
    Loads and displays a puzzle.
    """
    env, _ = load_puzzle(puzzle_name)
    print(f"Displaying puzzle: {puzzle_name}")
    print(env.render())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Pathery puzzle.")
    parser.add_argument(
        "puzzle_name",
        type=str,
        help="The name of the puzzle to visualize (e.g., puzzle_1).",
    )
    args = parser.parse_args()
    visualize_puzzle(args.puzzle_name)
