from typing import Optional
from pathery.solvers.genetic_solver import GeneticSolver
from pathery_env.envs.pathery import PatheryEnv, CellType
from pathery.rl.dqn_agent import DQNAgent


class HybridGASolver(GeneticSolver):
    """
    A solver that uses a hybrid genetic algorithm with a DQN agent.
    The mutation is performed by a DQN agent instead of being random.
    """

    def __init__(
        self,
        env: PatheryEnv,
        dqn_agent: Optional[DQNAgent] = None,
        disable_dqn_training: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the HybridGASolver.
        """
        super().__init__(env=env, **kwargs)
        self.dqn_agent = dqn_agent if dqn_agent is not None else DQNAgent()
        self.disable_dqn_training = disable_dqn_training

        if self.disable_dqn_training:
            self.dqn_agent.set_inference_mode()

    def _mutate(self, env: PatheryEnv, data_logger=None) -> PatheryEnv:
        """
        Performs a mutation on a chromosome using the DQN agent.
        """
        mutated_env = env.copy()
        pre_mutation_state = mutated_env.grid.copy()
        fitness_before = self._calculate_fitness(mutated_env)

        action = self.dqn_agent.choose_action(pre_mutation_state)

        if not action:
            return mutated_env

        if action["type"] == "MOVE":
            from_pos = action["from"]
            to_pos = action["to"]
            mutated_env.grid[from_pos[1], from_pos[0]] = CellType.OPEN.value
            mutated_env.grid[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "ADD":
            to_pos = action["to"]
            mutated_env.grid[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "REMOVE":
            from_pos = action["from"]
            mutated_env.grid[from_pos[1], from_pos[0]] = CellType.OPEN.value

        fitness_after = self._calculate_fitness(mutated_env)
        reward = fitness_after - fitness_before

        self.dqn_agent.replay_buffer.push(pre_mutation_state, action, reward)

        return mutated_env

    def _after_new_population_hook(self, generation: int):
        """
        After each generation, train the DQN agent and decay epsilon.
        """
        if not self.disable_dqn_training:
            self.dqn_agent.train_step()
            self.dqn_agent.decay_epsilon()
