import argparse
import os
import sys
import time
import pickle
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.rl.dqn_agent import DQNAgent
from src.pathery.utils import load_puzzle


def load_preprocessed_data(data_dir, start_time):
    """
    Loads preprocessed data from .pkl files into three lists based on reward.
    """
    print(f"[{time.time() - start_time:.2f}s] Starting to load preprocessed data...")
    experiences = {"positive": [], "zero": [], "negative": []}

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".pkl"):
            print(f"[{time.time() - start_time:.2f}s] Loading {filename}...")
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "rb") as f:
                batch_data = pickle.load(f)
                for entry in batch_data:
                    reward = entry["reward"]
                    if reward > 0:
                        experiences["positive"].append(entry)
                    elif reward == 0:
                        experiences["zero"].append(entry)
                    else:
                        experiences["negative"].append(entry)
            print(f"[{time.time() - start_time:.2f}s] Finished loading {filename}.")

    print(f"[{time.time() - start_time:.2f}s] Finished loading all preprocessed data.")
    total_entries = sum(len(v) for v in experiences.values())
    print(f"  Loaded {total_entries} entries.")
    for reward_type, data in experiences.items():
        print(f"  - {reward_type.capitalize()}: {len(data)} entries")
    return experiences


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
        "--load_model_path",
        type=str,
        default=None,
        help="Path to load the model checkpoint for fine-tuning.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Path to save the trained model checkpoint.",
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"[{time.time() - start_time:.2f}s] Starting script...")

    # Determine save path
    save_model_path = args.save_model_path
    if save_model_path is None:
        if args.load_model_path:
            save_model_path = f"{args.load_model_path}_finetuned"
        else:
            save_model_path = "output/checkpoints_new"

    # Construct absolute paths
    if args.load_model_path:
        args.load_model_path = os.path.abspath(args.load_model_path)
    save_model_path = os.path.abspath(save_model_path)

    # Create a dummy environment to initialize the agent
    env, _ = load_puzzle("data/puzzles/ucu/puzzle_6.json")

    agent = DQNAgent(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        model_path=args.load_model_path,
    )
    print(f"[{time.time() - start_time:.2f}s] DQN Agent created.")

    # Load the preprocessed data
    experiences = load_preprocessed_data(args.data_dir, start_time)
    print(f"[{time.time() - start_time:.2f}s] Data loaded.")

    writer = SummaryWriter()
    print(
        f"[{time.time() - start_time:.2f}s] SummaryWriter created. Starting training..."
    )

    # Pre-train the agent on the loaded dataset
    agent.pretrain(experiences, args.epochs, args.batch_size, start_time, writer=writer)
    print(f"[{time.time() - start_time:.2f}s] Training finished.")

    writer.close()

    # Save the trained model
    agent.save_model(save_model_path, args.epochs)
    print(f"[{time.time() - start_time:.2f}s] Model saved to {save_model_path}")
