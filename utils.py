import json
import os
import sys
from pathery_env.envs.pathery import PatheryEnv
from tests.map_builder import MapBuilder


def load_puzzle(puzzle_name_or_path: str):
    """
    Loads a puzzle from a name or path and returns the environment and best known solution.
    """
    puzzle_path = puzzle_name_or_path
    if not puzzle_path.endswith(".json"):
        puzzle_path = os.path.join("puzzles", f"{puzzle_name_or_path}.json")

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
