import heapq
import logging
import random
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from pathery_env.envs.pathery import CellType, PatheryEnv

from solvers.base_solver import BaseSolver


class FocusedSearchSolver(BaseSolver):
    """
    A solver that uses a focused beam search algorithm.
    """

    def __init__(
        self,
        env: PatheryEnv,
        beam_width: int = 10,
        search_depth: int = 5,
        best_known_solution: int = 0,
        time_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the FocusedSearchSolver.

        Args:
            env (PatheryEnv): An instance of the PatheryEnv.
            beam_width (int): The number of candidates to keep in the beam.
            search_depth (int): The depth of the search.
            best_known_solution (int): The best known solution length.
            time_limit (Optional[int]): The time limit in seconds for the solver.
        """
        super().__init__(
            env,
            best_known_solution,
            time_limit,
            kwargs.get("perf_logger"),
        )
        self.beam_width = beam_width
        self.search_depth = search_depth

    def solve(self) -> Tuple[Optional[List[Tuple[int, int]]], int]:
        """
        Attempts to find the longest path using a focused beam search.

        Returns:
            tuple: A tuple containing the best path found and its length.
        """
        self.start_time = time.time()
        self.env.reset()
        self._randomly_place_walls(self.env.wallsToPlace)

        initial_walls = np.where(self.env.grid == CellType.WALL.value)
        initial_walls = list(zip(initial_walls[1], initial_walls[0]))
        initial_path = self.env._calculateShortestPath()
        initial_length = len(initial_path)

        if not initial_path.any():
            return None, 0

        beam = [(initial_length, initial_walls)]
        best_length = initial_length
        best_walls = initial_walls

        for i in range(self.search_depth):
            if self.time_limit and (time.time() - self.start_time) > self.time_limit:
                print(f"Time limit reached. Exiting after {i} iterations.")
                break
            candidates = []
            for length, walls in beam:
                for _ in range(self.beam_width):  # Create beam_width mutations
                    new_walls = self._mutate(walls)
                    self.env.reset()
                    logging.info(
                        f"Stepping with {len(new_walls)} walls in the main loop"
                    )
                    for x, y in new_walls:
                        self._add_wall(x, y)
                    path = self.env._calculateShortestPath()
                    if path.any():
                        heapq.heappush(candidates, (-len(path), new_walls))

            beam = []
            seen_walls = set()
            while candidates and len(beam) < self.beam_width:
                length, walls = heapq.heappop(candidates)
                walls_tuple = tuple(sorted(walls))
                if walls_tuple not in seen_walls:
                    beam.append((-length, walls))
                    seen_walls.add(walls_tuple)

            if not beam:
                break

            current_best_length, current_best_walls = beam[0]

            if current_best_length > best_length:
                best_length = current_best_length
                best_walls = current_best_walls

        self.env.reset()
        logging.info(f"Stepping with {len(best_walls)} walls at the end")
        for x, y in best_walls:
            self._add_wall(x, y)
        best_path = self.env._calculateShortestPath()

        return best_path, best_length

    def _mutate(self, walls: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not walls:
            return []

        mutated_walls = walls.copy()
        wall_to_mutate = random.choice(mutated_walls)
        mutated_walls.remove(wall_to_mutate)

        empty_cells = np.where(self.env.grid == CellType.OPEN.value)
        empty_cells = list(zip(empty_cells[1], empty_cells[0]))

        if not empty_cells:
            return walls

        new_wall_pos = random.choice(empty_cells)
        mutated_walls.append(new_wall_pos)
        return mutated_walls
