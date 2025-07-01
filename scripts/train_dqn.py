import argparse
import json
import random
import os
import sys
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pathery.rl.dqn_agent import DQNAgent
from src.pathery.utils import load_puzzle


def create_balanced_dataset(data_path: str, oversample_factor: int = 10):
    """
    Creates a balanced dataset by oversampling positive rewards and undersampling others.
    """
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    positive_rewards = [d for d in data if d["reward"] > 0]
    zero_rewards = [d for d in data if d["reward"] == 0]
    negative_rewards = [d for d in data if d["reward"] < 0]

    # Oversample positive rewards
    balanced_data = positive_rewards * oversample_factor

    # Undersample zero and negative rewards to match the oversampled positive count
    num_samples = len(balanced_data)
    balanced_data.extend(
        random.sample(zero_rewards, k=min(num_samples, len(zero_rewards)))
    )
    balanced_data.extend(
        random.sample(negative_rewards, k=min(num_samples, len(negative_rewards)))
    )

    random.shuffle(balanced_data)
    return balanced_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("data_path", type=str, help="Path to the training data.")
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

    # Construct absolute path for model checkpoint
    args.model_path = os.path.abspath(args.model_path)

    # Create a dummy environment to initialize the agent
    env, _ = load_puzzle("data/puzzles/ucu/puzzle_6.json")

    agent = DQNAgent(
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    # Set up Orbax checkpointer
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(args.model_path, options=options)

    # Create a balanced dataset
    balanced_data = create_balanced_dataset(args.data_path)

    # Load the balanced data into the replay buffer
    for entry in balanced_data:
        state = entry["pre_mutation_state"]
        if len(state) != 19 or len(state[0]) != 27:
            print(f"Skipping state with unexpected shape: {len(state)}x{len(state[0])}")
            continue
        agent.replay_buffer.push(
            state,
            entry["mutation_info"],
            entry["reward"],
        )

    writer = SummaryWriter()

    # Pre-train the agent on the balanced dataset
    agent.pretrain(args.epochs, args.batch_size, writer=writer)

    writer.close()

    # Save the trained model
    mngr.save(args.epochs, args=ocp.args.StandardSave(agent.state))
    mngr.wait_until_finished()
    print(f"Model saved to {args.model_path}")
