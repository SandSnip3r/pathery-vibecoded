import argparse
import json
import os
import pickle
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def one_hot_encode(state: np.ndarray, max_channels: int) -> np.ndarray:
    """Converts a 2D grid state into a one-hot encoded 3D array."""
    one_hot = np.zeros((*state.shape, max_channels), dtype=np.float32)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            one_hot[i, j, state[i, j]] = 1
    return one_hot


def preprocess_logs(log_dir, output_dir, batch_size=100):
    """
    Converts JSONL log files to pickled format in batches, with validation.
    """
    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting preprocessing...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(
            f"[{time.time() - start_time:.2f}s] Created output directory: {output_dir}"
        )

    log_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
    total_files = len(log_files)
    processed_files = 0
    batch_num = 0
    valid_lines = 0
    invalid_shape_lines = 0

    print(
        f"[{time.time() - start_time:.2f}s] Found {total_files} log files to process."
    )

    for i in range(0, total_files, batch_size):
        batch_start_time = time.time()
        batch_files = log_files[i : i + batch_size]
        valid_data = []
        print(f"[{time.time() - start_time:.2f}s] Processing batch {batch_num + 1}...")

        for filename in batch_files:
            print(
                f"[{time.time() - start_time:.2f}s] Processing {filename} ({processed_files + 1}/{total_files})..."
            )
            file_path = os.path.join(log_dir, filename)
            with open(file_path, "r") as infile:
                for line in infile:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue
                    try:
                        data = json.loads(stripped_line)
                        state = data.get("pre_mutation_state")
                        if (
                            isinstance(state, list)
                            and len(state) == 19
                            and all(len(row) == 27 for row in state)
                        ):
                            valid_data.append(data)
                            valid_lines += 1
                        else:
                            invalid_shape_lines += 1
                    except (json.JSONDecodeError, TypeError):
                        continue
            processed_files += 1

        if valid_data:
            output_path = os.path.join(output_dir, f"batch_{batch_num}.pkl")
            with open(output_path, "wb") as outfile:
                pickle.dump(valid_data, outfile)
            print(
                f"[{time.time() - start_time:.2f}s] Saved batch {batch_num + 1} to {output_path}"
            )
            batch_num += 1

        print(
            f"[{time.time() - start_time:.2f}s] Finished batch {batch_num}. "
            f"Batch took {time.time() - batch_start_time:.2f}s."
        )

    print(f"[{time.time() - start_time:.2f}s] Preprocessing complete.")
    print(f"  - Valid lines: {valid_lines}")
    print(f"  - Invalid shape lines: {invalid_shape_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GA transition logs.")
    parser.add_argument(
        "log_dir", type=str, help="Directory containing the raw log files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save the processed data."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of log files to process in each batch.",
    )
    args = parser.parse_args()

    preprocess_logs(args.log_dir, args.output_dir, args.batch_size)
