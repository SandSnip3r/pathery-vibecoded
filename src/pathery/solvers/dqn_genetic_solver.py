import logging
import os
from typing import Optional
from pathery_env.envs.pathery import PatheryEnv, CellType
from pathery.solvers.genetic_solver import GeneticSolver
import numpy as np
import orbax.checkpoint as ocp
from pathery.rl.dqn_agent import DQNAgent


class DqnGeneticSolver(GeneticSolver):
    """
    A genetic algorithm solver that uses a DQN to guide mutations.
    """

    def __init__(
        self,
        env: PatheryEnv,
        population_size: int = 100,
        generations: int = 100,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        perf_logger: Optional[logging.Logger] = None,
        data_log_dir: Optional[str] = None,
        model_path: str = "output/checkpoints",
        **kwargs,
    ) -> None:
        """
        Initializes the DqnGeneticSolver.
        """
        super().__init__(
            env,
            population_size,
            generations,
            tournament_size,
            mutation_rate,
            crossover_rate,
            best_known_solution,
            time_limit,
            perf_logger,
            data_log_dir,
            **kwargs,
        )
        self.agent = DQNAgent(env)
        model_path = os.path.abspath(model_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
        mngr = ocp.CheckpointManager(model_path, options=options)
        latest_step = mngr.latest_step()
        if latest_step is not None:
            self.agent.state = mngr.restore(
                latest_step, args=ocp.args.StandardRestore(self.agent.state)
            )

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Performs a mutation on a chromosome using the DQN model.
        """
        mutated_chromosome = chromosome.copy()
        action = self.agent.choose_action(mutated_chromosome, epsilon=0.0)

        if action["type"] == "MOVE":
            from_pos = action["from"]
            to_pos = action["to"]
            mutated_chromosome[from_pos[1], from_pos[0]] = CellType.OPEN.value
            mutated_chromosome[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "ADD":
            to_pos = action["to"]
            mutated_chromosome[to_pos[1], to_pos[0]] = CellType.WALL.value
        elif action["type"] == "REMOVE":
            from_pos = action["from"]
            mutated_chromosome[from_pos[1], from_pos[0]] = CellType.OPEN.value

        return mutated_chromosome
