import os
import json
from typing import Optional, Any
from pathery_env.envs.pathery import PatheryEnv
from pathery.solvers.base_solver import BaseSolver
from pathery.solvers.genetic_solver import GeneticSolver
import orbax.checkpoint as ocp
from pathery.rl.dqn_agent import DQNAgent


class DqnGeneticSolver(GeneticSolver):
    """
    A genetic algorithm solver that uses a DQN to guide mutations.
    """

    def __init__(
        self,
        env: PatheryEnv,
        model_path: str = "output/checkpoints",
        epsilon: float = 0.15,
        **kwargs,
    ) -> None:
        """
        Initializes the DqnGeneticSolver.
        """
        super().__init__(env, **kwargs)
        self.agent = DQNAgent()
        self.epsilon = epsilon
        model_path = os.path.abspath(model_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        mngr = ocp.CheckpointManager(model_path, options=options)
        latest_step = mngr.latest_step()
        if latest_step is not None:
            self.agent.state = mngr.restore(
                latest_step, args=ocp.args.StandardRestore(self.agent.state)
            )

    def _mutate(self, env: PatheryEnv, data_logger: Optional[Any] = None) -> PatheryEnv:
        """
        Performs a mutation on a chromosome using the DQN model.
        """
        mutated_env = env.copy()
        pre_mutation_state = mutated_env.grid.copy()
        fitness_before = self._calculate_fitness(mutated_env)

        action = self.agent.choose_action(mutated_env.grid, self.epsilon)

        if not action:
            return mutated_env

        if action["type"] == "MOVE":
            from_pos = action["from"]
            to_pos = action["to"]
            BaseSolver._remove_wall(mutated_env, from_pos[0], from_pos[1])
            BaseSolver._add_wall(mutated_env, to_pos[0], to_pos[1])
        elif action["type"] == "ADD":
            to_pos = action["to"]
            BaseSolver._add_wall(mutated_env, to_pos[0], to_pos[1])
        elif action["type"] == "REMOVE":
            from_pos = action["from"]
            BaseSolver._remove_wall(mutated_env, from_pos[0], from_pos[1])

        fitness_after = self._calculate_fitness(mutated_env)
        reward = fitness_after - fitness_before

        if data_logger:
            log_entry = {
                "pre_mutation_state": pre_mutation_state.tolist(),
                "mutation_info": action,
                "reward": reward,
            }
            data_logger.write(json.dumps(log_entry) + "\n")
            data_logger.flush()

        return mutated_env
