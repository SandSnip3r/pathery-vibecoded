import os
import argparse
import json


def combine_logs(log_dir, output_file):
    """
    Combines all .jsonl files in a directory into a single, validated
    JSONL file. It filters out any invalid or incomplete JSON lines, and any
    entries that do not have the expected 27x19 shape.
    """
    total_lines = 0
    valid_lines = 0
    corrupt_lines = 0
    invalid_shape_lines = 0

    output_filename = os.path.basename(output_file)

    with open(output_file, "w") as outfile:
        for filename in os.listdir(log_dir):
            if filename.endswith(".jsonl") and filename != output_filename:
                file_path = os.path.join(log_dir, filename)
                with open(file_path, "r") as infile:
                    for line in infile:
                        total_lines += 1
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue  # Skip empty lines
                        try:
                            data = json.loads(stripped_line)

                            # Validate the shape
                            state = data.get("pre_mutation_state")
                            if (
                                isinstance(state, list)
                                and len(state) == 19
                                and all(len(row) == 27 for row in state)
                            ):
                                outfile.write(stripped_line + "\n")
                                valid_lines += 1
                            else:
                                invalid_shape_lines += 1

                        except (json.JSONDecodeError, TypeError):
                            # This line is corrupt, so we skip it.
                            corrupt_lines += 1
                            continue

    print("Log combination summary:")
    print(f"  Total lines processed: {total_lines}")
    print(f"  Valid lines written: {valid_lines}")
    print(f"  Skipped (corrupt): {corrupt_lines}")
    print(f"  Skipped (invalid shape): {invalid_shape_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine mutation logs.")
    parser.add_argument(
        "log_dir", type=str, help="The directory containing the log files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file.",
    )
    args = parser.parse_args()

    combine_logs(args.log_dir, args.output_file)
