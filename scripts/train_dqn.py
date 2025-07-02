import argparse
import os
import sys
import time
import pickle
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.rl.dqn_agent import DQNAgent
from src.pathery.utils import load_puzzle


def load_preprocessed_data(data_dir, agent, start_time):
    """
    Loads preprocessed data from .pkl files into the replay buffer.
    """
    print(f"[{time.time() - start_time:.2f}s] Starting to load preprocessed data...")
    total_entries = 0

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".pkl"):
            print(f"[{time.time() - start_time:.2f}s] Loading {filename}...")
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "rb") as f:
                batch_data = pickle.load(f)
                for entry in batch_data:
                    agent.replay_buffer.push(
                        entry["pre_mutation_state"],
                        entry["mutation_info"],
                        entry["reward"],
                    )
                    total_entries += 1
            print(f"[{time.time() - start_time:.2f}s] Finished loading {filename}.")

    print(f"[{time.time() - start_time:.2f}s] Finished loading all preprocessed data.")
    print(f"  Loaded {total_entries} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the preprocessed training data.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/checkpoints",
        help="Path to save the trained model checkpoint.",
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting script...")

    # Construct absolute path for model checkpoint
    args.model_path = os.path.abspath(args.model_path)

    # Create a dummy environment to initialize the agent
    env, _ = load_puzzle("data/puzzles/ucu/puzzle_6.json")

    agent = DQNAgent(
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    print(f"[{time.time() - start_time:.2f}s] DQN Agent created.")

    # Set up Orbax checkpointer
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(args.model_path, options=options)
    print(f"[{time.time() - start_time:.2f}s] Checkpoint manager created.")

    # Load the preprocessed data into the replay buffer
    load_preprocessed_data(args.data_dir, agent, start_time)
    agent.replay_buffer.print_buffer_sizes()
    print(f"[{time.time() - start_time:.2f}s] Replay buffer filled.")

    writer = SummaryWriter()
    print(
        f"[{time.time() - start_time:.2f}s] SummaryWriter created. Starting training..."
    )

    # Pre-train the agent on the loaded dataset
    agent.pretrain(args.epochs, args.batch_size, writer=writer)
    print(f"[{time.time() - start_time:.2f}s] Training finished.")

    writer.close()

    # Save the trained model
    mngr.save(args.epochs, args=ocp.args.StandardSave(agent.state))
    mngr.wait_until_finished()
    print(f"[{time.time() - start_time:.2f}s] Model saved to {args.model_path}")
