import json
import os
import sys

from pathery_env.envs.pathery import PatheryEnv

from pathery.map_builder import MapBuilder


def load_puzzle(puzzle_name_or_path: str):
    """
    Loads a puzzle from a name or path and returns the environment and best known solution.
    """
    # If an absolute path that exists is given, use it.
    if os.path.isabs(puzzle_name_or_path) and os.path.exists(puzzle_name_or_path):
        puzzle_path = puzzle_name_or_path
    else:
        # Build a base path to the `data/puzzles` directory.
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        puzzles_dir = os.path.join(project_root, "data", "puzzles")

        # Normalize the puzzle name (add .json if missing)
        puzzle_filename = (
            puzzle_name_or_path
            if puzzle_name_or_path.endswith(".json")
            else f"{puzzle_name_or_path}.json"
        )

        # First, check if the path is a direct relative path from the puzzles dir
        potential_path = os.path.join(puzzles_dir, puzzle_name_or_path)
        if not potential_path.endswith(".json"):
            potential_path += ".json"

        if os.path.exists(potential_path):
            puzzle_path = potential_path
        else:
            # If not, search recursively for the filename
            found_path = None
            for root, _, files in os.walk(puzzles_dir):
                if os.path.basename(puzzle_filename) in files:
                    found_path = os.path.join(root, os.path.basename(puzzle_filename))
                    break
            if found_path:
                puzzle_path = found_path
            else:
                print(
                    f"[bold red]Error: Puzzle file '{puzzle_filename}' not found in '{puzzles_dir}' or its subdirectories.[/bold red]"
                )
                sys.exit(1)

    if not os.path.exists(puzzle_path):
        print(f"[bold red]Error: Puzzle file not found at {puzzle_path}[/bold red]")
        sys.exit(1)

    with open(puzzle_path, "r") as f:
        puzzle_data = json.load(f)

    if "map_string" in puzzle_data:
        env = PatheryEnv.fromMapString(
            render_mode="ansi", map_string=puzzle_data["map_string"]
        )
    else:
        builder = MapBuilder(
            puzzle_data["width"], puzzle_data["height"], puzzle_data["num_walls"]
        )
        builder.set_start(puzzle_data["start"][0], puzzle_data["start"][1])
        builder.set_finish(puzzle_data["finish"][0], puzzle_data["finish"][1])

        for rock in puzzle_data["rocks"]:
            builder.add_rock(rock[0], rock[1])

        if "checkpoints" in puzzle_data:
            for checkpoint in puzzle_data["checkpoints"]:
                builder.add_checkpoint(checkpoint[0], checkpoint[1], checkpoint[2])

        env = PatheryEnv(render_mode="ansi", map_string=builder.build())

    env.reset()
    return env, puzzle_data
