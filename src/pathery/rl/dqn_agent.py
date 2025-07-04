import random
import os
import time
from typing import Any, Dict, Optional

from flax.training import checkpoints
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tensorboardX import SummaryWriter
import orbax.checkpoint as ocp

from pathery_env.envs.pathery import CellType


class ExperienceReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, key: jax.random.PRNGKey):
        self.capacity = capacity // 3
        self.batch_size = batch_size
        self.key = key
        self.buffers = {
            "positive": self._create_buffer(),
            "zero": self._create_buffer(),
            "negative": self._create_buffer(),
        }

    def _create_buffer(self):
        return {
            "states": np.zeros(
                (self.capacity, 19, 27),
                dtype=np.int32,
            ),
            "action_types": np.zeros((self.capacity, 1), dtype=np.int32),
            "from_pos": np.zeros((self.capacity, 2), dtype=np.int32),
            "to_pos": np.zeros((self.capacity, 2), dtype=np.int32),
            "rewards": np.zeros((self.capacity, 1), dtype=np.float32),
            "position": 0,
            "size": 0,
        }

    def _one_hot_encode(self, state: np.ndarray, max_channels: int) -> np.ndarray:
        """Converts a 2D grid state into a one-hot encoded 3D array."""
        one_hot = np.zeros((*state.shape, max_channels), dtype=np.float32)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                one_hot[i, j, state[i, j]] = 1
        return one_hot

    def push(self, state, action, reward):
        if reward > 0:
            buffer = self.buffers["positive"]
        elif reward == 0:
            buffer = self.buffers["zero"]
        else:
            buffer = self.buffers["negative"]

        position = buffer["position"]
        buffer["states"][position] = np.array(state)

        action_type_map = {"MOVE": 0, "ADD": 1, "REMOVE": 2}
        buffer["action_types"][position] = action_type_map[action["type"]]
        buffer["from_pos"][position] = action.get("from", [-1, -1])
        buffer["to_pos"][position] = action.get("to", [-1, -1])
        buffer["rewards"][position] = reward

        buffer["position"] = (position + 1) % self.capacity
        buffer["size"] = min(buffer["size"] + 1, self.capacity)

    def sample(self):
        self.key, subkey = jax.random.split(self.key)
        batch_size_per_buffer = self.batch_size // 3

        samples = []
        for buffer_type in ["positive", "zero", "negative"]:
            buffer = self.buffers[buffer_type]
            if buffer["size"] > 0:
                indices = jax.random.randint(
                    subkey, (batch_size_per_buffer,), 0, buffer["size"]
                )
                batch = {
                    k: v[indices]
                    for k, v in buffer.items()
                    if k not in ["position", "size"]
                }
                samples.append(batch)

        # Concatenate samples from all buffers
        if not samples:
            return {}

        concatenated_batch = {}
        for key in samples[0].keys():
            concatenated_batch[key] = np.concatenate([s[key] for s in samples], axis=0)

        return concatenated_batch

    def print_buffer_sizes(self):
        print("Replay buffer sizes:")
        for buffer_type, buffer in self.buffers.items():
            print(f"  - {buffer_type.capitalize()}: {buffer['size']} items")

    def __len__(self):
        return sum(b["size"] for b in self.buffers.values())


class DQN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Input: (batch_size, height, width, channels)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        # Output heads
        removal_scores = nn.Dense(features=27 * 19, name="removal")(x)
        placement_scores = nn.Dense(features=27 * 19, name="placement")(x)
        action_type_scores = nn.Dense(features=3, name="action_type")(
            x
        )  # MOVE, ADD, REMOVE

        return removal_scores, placement_scores, action_type_scores


class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        y = nn.relu(y)
        y = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(y)
        return nn.relu(x + y)


class AttentionBlock(nn.Module):
    features: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        # Simple self-attention
        batch_size, height, width, channels = x.shape
        x_reshaped = x.reshape(batch_size, height * width, channels)

        attended_x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.features
        )(x_reshaped, x_reshaped)

        attended_x = attended_x.reshape(batch_size, height, width, channels)

        return nn.LayerNorm()(x + attended_x)


class DQN_ResNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        for _ in range(4):
            x = ResidualBlock(features=64)(x)
        x = AttentionBlock(features=64)(x)
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        # Output heads
        removal_scores = nn.Dense(features=27 * 19, name="removal")(x)
        placement_scores = nn.Dense(features=27 * 19, name="placement")(x)
        action_type_scores = nn.Dense(features=3, name="action_type")(
            x
        )  # MOVE, ADD, REMOVE

        return removal_scores, placement_scores, action_type_scores


class DQNAgent:
    def __init__(
        self,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        grid_size=(19, 27),
        max_channels=34,
        model_path: Optional[str] = None,
    ):
        self.grid_size = grid_size
        self.max_channels = max_channels
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(0)
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size, self.key)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = DQN_ResNet()

        dummy_state = jnp.zeros(
            (1, self.grid_size[0], self.grid_size[1], self.max_channels)
        )
        params = self.model.init(self.key, dummy_state)["params"]

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
        )

        # Restore from the latest checkpoint if a path is provided
        if model_path:
            model_path = os.path.abspath(model_path)
            self.state = checkpoints.restore_checkpoint(model_path, self.state)

    def set_inference_mode(self):
        """Sets epsilon to 0 for pure exploitation."""
        self.epsilon = 0.0

    def decay_epsilon(self):
        """Decays epsilon for exploration-exploitation trade-off."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def _one_hot_encode(self, state: np.ndarray) -> np.ndarray:
        """Converts a 2D grid state into a one-hot encoded 3D array."""
        one_hot = jax.nn.one_hot(state, self.max_channels)
        return jnp.transpose(one_hot, (1, 0, 2))

    def choose_action(self, board_state: np.ndarray, epsilon: float) -> Dict[str, Any]:
        """
        Chooses an action based on the board state, ensuring validity.
        """
        wall_positions = np.argwhere(board_state == CellType.WALL.value)
        open_squares = np.argwhere(board_state == CellType.OPEN.value)

        possible_actions = []
        if len(open_squares) > 0:
            possible_actions.append("ADD")
        if len(wall_positions) > 0:
            possible_actions.append("REMOVE")
        if len(wall_positions) > 0 and len(open_squares) > 0:
            possible_actions.append("MOVE")

        if not possible_actions:
            return {}

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_type = random.choice(possible_actions)
            if action_type == "ADD":
                pos = random.choice(open_squares)
                return {"type": "ADD", "to": [int(pos[1]), int(pos[0])]}
            elif action_type == "REMOVE":
                pos = random.choice(wall_positions)
                return {"type": "REMOVE", "from": [int(pos[1]), int(pos[0])]}
            elif action_type == "MOVE":
                from_pos = random.choice(wall_positions)
                to_pos = random.choice(open_squares)
                return {
                    "type": "MOVE",
                    "from": [int(from_pos[1]), int(from_pos[0])],
                    "to": [int(to_pos[1]), int(to_pos[0])],
                }
        else:
            # Greedy action with masking
            obs = self._one_hot_encode(board_state)
            obs = jnp.expand_dims(obs, axis=0)

            removal_scores, placement_scores, action_type_scores = self.state.apply_fn(
                {"params": self.state.params}, obs
            )

            # Mask invalid actions
            mask_add = "ADD" in possible_actions
            mask_remove = "REMOVE" in possible_actions
            mask_move = "MOVE" in possible_actions
            action_type_scores = jnp.where(
                jnp.array([mask_move, mask_add, mask_remove]),
                action_type_scores,
                -jnp.inf,
            )

            # Mask invalid positions
            wall_mask = (board_state == CellType.WALL.value).flatten()
            open_mask = (board_state == CellType.OPEN.value).flatten()
            removal_scores = jnp.where(wall_mask, removal_scores, -jnp.inf)
            placement_scores = jnp.where(open_mask, placement_scores, -jnp.inf)

            action_type = jnp.argmax(action_type_scores).item()

            if action_type == 0:  # MOVE
                from_pos_flat = jnp.argmax(removal_scores).item()
                to_pos_flat = jnp.argmax(placement_scores).item()
                from_pos = [
                    int(from_pos_flat % self.grid_size[1]),
                    int(from_pos_flat // self.grid_size[1]),
                ]
                to_pos = [
                    int(to_pos_flat % self.grid_size[1]),
                    int(to_pos_flat // self.grid_size[1]),
                ]
                return {"type": "MOVE", "from": from_pos, "to": to_pos}
            elif action_type == 1:  # ADD
                to_pos_flat = jnp.argmax(placement_scores).item()
                to_pos = [
                    int(to_pos_flat % self.grid_size[1]),
                    int(to_pos_flat // self.grid_size[1]),
                ]
                return {"type": "ADD", "to": to_pos}
            elif action_type == 2:  # REMOVE
                from_pos_flat = jnp.argmax(removal_scores).item()
                from_pos = [
                    int(from_pos_flat % self.grid_size[1]),
                    int(from_pos_flat // self.grid_size[1]),
                ]
                return {"type": "REMOVE", "from": from_pos}

        return {}

    @staticmethod
    @jax.jit
    def _jitted_train_step(
        state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ):
        rewards = batch["rewards"]
        action_types = batch["action_types"]
        from_pos = batch["from_pos"]
        to_pos = batch["to_pos"]

        def loss_fn(params):
            # One-hot encode the states inside the jitted function
            states_one_hot = jax.nn.one_hot(batch["states"], num_classes=34)
            # Transpose to (batch, height, width, channels)
            states_one_hot = jnp.transpose(states_one_hot, (0, 2, 1, 3))

            removal_scores, placement_scores, action_type_scores = state.apply_fn(
                {"params": params}, states_one_hot
            )

            def q_value_fn(i):
                from_idx = from_pos[i, 1] * 27 + from_pos[i, 0]
                to_idx = to_pos[i, 1] * 27 + to_pos[i, 0]

                q_move = (
                    action_type_scores[i, 0]
                    + removal_scores[i, from_idx]
                    + placement_scores[i, to_idx]
                )
                q_add = action_type_scores[i, 1] + placement_scores[i, to_idx]
                q_remove = action_type_scores[i, 2] + removal_scores[i, from_idx]

                return jax.lax.switch(
                    action_types[i, 0],
                    [lambda: q_move, lambda: q_add, lambda: q_remove],
                )

            q_values = jax.vmap(q_value_fn)(jnp.arange(len(rewards)))
            return jnp.mean((q_values - rewards.squeeze()) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def save_model(self, save_path: str, epoch: int):
        """Saves the model checkpoint."""
        save_path = os.path.abspath(save_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        mngr = ocp.CheckpointManager(save_path, options=options)
        mngr.save(epoch, args=ocp.args.StandardSave(self.state))
        mngr.wait_until_finished()
        print(f"Model saved to {save_path}")

    def train_step(self):
        """
        Performs a single training step on a batch from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample()
        if not batch:
            return

        self.state, loss = self._jitted_train_step(self.state, batch)
        return loss

    def pretrain(
        self,
        experiences: Dict[str, list],
        epochs: int,
        batch_size: int,
        start_time: float,
        writer: Optional[SummaryWriter] = None,
    ):
        # Internal backup manager
        backup_model_path = os.path.abspath(
            os.path.join(
                "output", f"checkpoints_backup_{time.strftime('%Y%m%d-%H%M%S')}"
            )
        )
        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        backup_mngr = ocp.CheckpointManager(backup_model_path, options=options)

        # Calculate number of batches based on the smallest category
        min_len = min(len(v) for v in experiences.values())
        num_batches = min_len // (batch_size // 3)

        for epoch in range(epochs):
            # Shuffle each experience list
            for key in experiences:
                np.random.shuffle(experiences[key])

            total_loss = 0
            for batch_idx in range(num_batches):
                batch_data = []
                batch_size_per_buffer = batch_size // 3
                for key in ["positive", "zero", "negative"]:
                    start = batch_idx * batch_size_per_buffer
                    end = start + batch_size_per_buffer
                    batch_data.extend(experiences[key][start:end])

                # Convert list of dicts to dict of lists
                batch = {
                    "states": np.array([e["pre_mutation_state"] for e in batch_data]),
                    "action_types": np.array(
                        [
                            {"MOVE": 0, "ADD": 1, "REMOVE": 2}[
                                e["mutation_info"]["type"]
                            ]
                            for e in batch_data
                        ]
                    ).reshape(-1, 1),
                    "from_pos": np.array(
                        [e["mutation_info"].get("from", [-1, -1]) for e in batch_data]
                    ),
                    "to_pos": np.array(
                        [e["mutation_info"].get("to", [-1, -1]) for e in batch_data]
                    ),
                    "rewards": np.array([e["reward"] for e in batch_data]).reshape(
                        -1, 1
                    ),
                }

                self.state, loss = self._jitted_train_step(self.state, batch)

                total_loss += loss
                if writer:
                    global_step = epoch * num_batches + batch_idx
                    writer.add_scalar("Loss/train", loss, global_step)

            avg_loss = total_loss / num_batches
            print(
                f"[{time.time() - start_time:.2f}s] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}"
            )
            if writer:
                writer.add_scalar("Loss/epoch", avg_loss, epoch)

            # Save a backup checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                backup_mngr.save(epoch + 1, args=ocp.args.StandardSave(self.state))
                backup_mngr.wait_until_finished()
                print(
                    f"[{time.time() - start_time:.2f}s] Saved backup checkpoint at epoch {epoch + 1}"
                )
