import json
import argparse


def validate_jsonl(file_path):
    """
    Validates a JSONL file, checking for valid JSON and correct board dimensions.
    """
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            try:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                data = json.loads(stripped_line)

                # Check for the existence of the state key
                if "pre_mutation_state" not in data:
                    print(f"Warning on line {i+1}: 'pre_mutation_state' key not found.")
                    continue

                state = data["pre_mutation_state"]

                # Validate the shape
                if not isinstance(state, list) or not all(
                    isinstance(row, list) for row in state
                ):
                    print(
                        f"Error on line {i+1}: 'pre_mutation_state' is not a list of lists."
                    )
                    continue

                height = len(state)
                # Check if state is not empty to avoid index error on width
                if height > 0:
                    width = len(state[0])
                    if height != 19 or width != 27:
                        print(
                            f"Shape error on line {i+1}: Found shape {height}x{width}, expected 19x27."
                        )
                else:
                    print(f"Shape error on line {i+1}: Found empty state array.")

            except json.JSONDecodeError as e:
                print(f"JSON error on line {i+1}: {e}")
                return False
    print(f"Finished validation for {file_path}.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a JSONL file.")
    parser.add_argument("file_path", type=str, help="The path to the JSONL file.")
    args = parser.parse_args()

    validate_jsonl(args.file_path)
