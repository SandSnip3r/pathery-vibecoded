import argparse
import json

from pathery_env_adapter import PatheryEnvAdapter as PatheryEmulator


def visualize_puzzle(puzzle_name: str):
    """
    Loads and displays a puzzle.
    """
    try:
        with open(f"puzzles/{puzzle_name}.json", "r") as f:
            puzzle_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Puzzle '{puzzle_name}' not found.")
        return

    emulator = PatheryEmulator(
        puzzle_data["width"], puzzle_data["height"], puzzle_data["num_walls"]
    )

    emulator.set_start(puzzle_data["start"][0], puzzle_data["start"][1])
    emulator.set_finish(puzzle_data["finish"][0], puzzle_data["finish"][1])

    if "rocks" in puzzle_data:
        for rock in puzzle_data["rocks"]:
            emulator.add_rock(rock[0], rock[1])

    if "checkpoints" in puzzle_data:
        for checkpoint in puzzle_data["checkpoints"]:
            emulator.add_checkpoint(checkpoint[0], checkpoint[1], checkpoint[2])

    print(f"Displaying puzzle: {puzzle_name}")
    emulator.display()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a Pathery puzzle.")
    parser.add_argument(
        "puzzle_name",
        type=str,
        help="The name of the puzzle to visualize (e.g., puzzle_1).",
    )
    args = parser.parse_args()
    visualize_puzzle(args.puzzle_name)
