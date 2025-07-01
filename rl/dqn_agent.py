import random
from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tensorboardX import SummaryWriter

from pathery_env.envs.pathery import CellType, PatheryEnv


class ExperienceReplayBuffer:
    def __init__(self, capacity: int, batch_size: int, key: jax.random.PRNGKey):
        self.capacity = capacity
        self.batch_size = batch_size
        self.key = key
        self.buffers = {}

    def _one_hot_encode(self, state: np.ndarray, max_channels: int) -> np.ndarray:
        """Converts a 2D grid state into a one-hot encoded 3D array."""
        one_hot = np.zeros((*state.shape, max_channels), dtype=np.float32)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                one_hot[i, j, state[i, j]] = 1
        return one_hot

    def push(self, state, action, reward):
        if not self.buffers:
            # Initialize buffers on first push
            state_array = self._one_hot_encode(np.array(state), 34)
            state_array = np.transpose(state_array, (1, 0, 2))
            self.buffers = {
                "states": np.zeros(
                    (self.capacity, *state_array.shape), dtype=np.float32
                ),
                "action_types": np.zeros((self.capacity, 1), dtype=np.int32),
                "from_pos": np.zeros((self.capacity, 2), dtype=np.int32),
                "to_pos": np.zeros((self.capacity, 2), dtype=np.int32),
                "rewards": np.zeros((self.capacity, 1), dtype=np.float32),
            }
            self.position = 0
            self.size = 0

        state_array = self._one_hot_encode(np.array(state), 34)
        state_array = np.transpose(state_array, (1, 0, 2))
        self.buffers["states"][self.position] = state_array

        action_type_map = {"MOVE": 0, "ADD": 1, "REMOVE": 2}
        self.buffers["action_types"][self.position] = action_type_map[action["type"]]
        self.buffers["from_pos"][self.position] = action.get("from", [-1, -1])
        self.buffers["to_pos"][self.position] = action.get("to", [-1, -1])
        self.buffers["rewards"][self.position] = reward

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        self.key, subkey = jax.random.split(self.key)
        indices = jax.random.randint(subkey, (self.batch_size,), 0, self.size)
        return {k: v[indices] for k, v in self.buffers.items()}

    def __len__(self):
        return self.size


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


class DQNAgent:
    def __init__(
        self, env: PatheryEnv, learning_rate=1e-4, buffer_size=100000, batch_size=128
    ):
        self.env = env
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(0)
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, batch_size, self.key)

        self.model = DQN()

        # The number of channels is fixed to the maximum possible value
        # to ensure the network can handle any puzzle.
        MAX_CHANNELS = 34
        dummy_state = jnp.zeros(
            (1, self.env.gridSize[0], self.env.gridSize[1], MAX_CHANNELS)
        )
        params = self.model.init(self.key, dummy_state)["params"]

        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(learning_rate),
        )

    def choose_action(self, board_state: np.ndarray, epsilon: float) -> Dict[str, Any]:
        if random.random() < epsilon:
            # Random action
            mutation_type = random.choice(["MOVE", "ADD", "REMOVE"])
            if mutation_type == "MOVE":
                wall_positions = np.where(board_state == CellType.WALL.value)
                wall_positions = list(zip(wall_positions[1], wall_positions[0]))
                empty_squares = np.where(board_state == CellType.OPEN.value)
                empty_squares = list(zip(empty_squares[1], empty_squares[0]))
                if not wall_positions or not empty_squares:
                    return self.choose_action(board_state, epsilon)  # Try again
                wall_to_move = random.choice(wall_positions)
                new_position = random.choice(empty_squares)
                return {
                    "type": "MOVE",
                    "from": [int(wall_to_move[0]), int(wall_to_move[1])],
                    "to": [int(new_position[0]), int(new_position[1])],
                }
            elif mutation_type == "ADD":
                empty_squares = np.where(board_state == CellType.OPEN.value)
                empty_squares = list(zip(empty_squares[1], empty_squares[0]))
                if not empty_squares:
                    return self.choose_action(board_state, epsilon)  # Try again
                new_wall_position = random.choice(empty_squares)
                return {
                    "type": "ADD",
                    "to": [int(new_wall_position[0]), int(new_wall_position[1])],
                }
            elif mutation_type == "REMOVE":
                wall_positions = np.where(board_state == CellType.WALL.value)
                wall_positions = list(zip(wall_positions[1], wall_positions[0]))
                if not wall_positions:
                    return self.choose_action(board_state, epsilon)  # Try again
                wall_to_remove = random.choice(wall_positions)
                return {
                    "type": "REMOVE",
                    "from": [int(wall_to_remove[0]), int(wall_to_remove[1])],
                }
        else:
            # Greedy action
            # One-hot encode the board state and add a batch dimension
            obs = self.env._get_obs()["board"]
            obs = jnp.transpose(obs, (1, 2, 0))
            obs = jnp.expand_dims(obs, axis=0)

            removal_scores, placement_scores, action_type_scores = self.state.apply_fn(
                {"params": self.state.params}, obs
            )

            action_type = jnp.argmax(action_type_scores).item()

            if action_type == 0:  # MOVE
                removal_pos_flat = jnp.argmax(removal_scores).item()
                placement_pos_flat = jnp.argmax(placement_scores).item()
                from_pos = [int(removal_pos_flat % 27), int(removal_pos_flat // 27)]
                to_pos = [int(placement_pos_flat % 27), int(placement_pos_flat // 27)]
                return {"type": "MOVE", "from": from_pos, "to": to_pos}
            elif action_type == 1:  # ADD
                placement_pos_flat = jnp.argmax(placement_scores).item()
                to_pos = [int(placement_pos_flat % 27), int(placement_pos_flat // 27)]
                return {"type": "ADD", "to": to_pos}
            elif action_type == 2:  # REMOVE
                removal_pos_flat = jnp.argmax(removal_scores).item()
                from_pos = [int(removal_pos_flat % 27), int(removal_pos_flat // 27)]
                return {"type": "REMOVE", "from": from_pos}

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
            removal_scores, placement_scores, action_type_scores = state.apply_fn(
                {"params": params}, batch["states"]
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

    def pretrain(
        self,
        epochs: int,
        batch_size: int,
        writer: Optional[SummaryWriter] = None,
    ):
        num_batches = len(self.replay_buffer) // batch_size
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx in range(num_batches):
                batch = self.replay_buffer.sample()
                self.state, loss = self._jitted_train_step(self.state, batch)

                total_loss += loss
                if writer:
                    global_step = epoch * num_batches + batch_idx
                    writer.add_scalar("Loss/train", loss, global_step)

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
            if writer:
                writer.add_scalar("Loss/epoch", avg_loss, epoch)
