import requests
import json
import argparse
import os
import glob


def download_puzzle(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading puzzle: {e}")
        return None


def save_puzzle(puzzle_data, file_path):
    try:
        reformatted_data = {
            "map_string": puzzle_data.get("code", ""),
            "best_solution": 0,
        }
        with open(file_path, "w") as f:
            json.dump(reformatted_data, f, indent=4)
        print(f"Saved puzzle to {file_path}")
    except IOError as e:
        print(f"Error saving puzzle: {e}")


def get_highest_puzzle_number(paths):
    highest_num = 0
    for path in paths:
        puzzle_files = glob.glob(os.path.join(path, "puzzle_*.json"))
        for f in puzzle_files:
            try:
                num = int(os.path.basename(f).split("_")[1].split(".")[0])
                if num > highest_num:
                    highest_num = num
            except (ValueError, IndexError):
                continue
    return highest_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Pathery puzzles.")
    parser.add_argument(
        "num_puzzles", type=int, help="The number of puzzles to download."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/puzzles/ucu",
        help="The directory to save the puzzles in.",
    )
    args = parser.parse_args()

    url = "https://www.pathery.com/mapeditor?mapBySpecial=ultra%20complex%20unlimited"

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find the highest existing puzzle number from all relevant directories
    search_paths = ["data/puzzles", "data/puzzles/ucu"]
    highest_puzzle_num = get_highest_puzzle_number(search_paths)
    start_index = highest_puzzle_num + 1

    for i in range(args.num_puzzles):
        puzzle_data = download_puzzle(url)
        if puzzle_data:
            file_path = os.path.join(args.output_dir, f"puzzle_{i + start_index}.json")
            save_puzzle(puzzle_data, file_path)
