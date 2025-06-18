import gymnasium as gym
from pathery_env.envs.pathery import PatheryEnv, CellType
import numpy as np

class PatheryEnvAdapter:
    def __init__(self, width: int, height: int, num_walls: int) -> None:
        self.env = PatheryEnv(render_mode=None, map_string=f'{width}.{height}.{num_walls}.:')
        self.env.reset()
        self.width = width
        self.height = height
        self.num_walls = num_walls
        self.grid = self.env.grid

    def get_wall_locations(self):
        wall_locations = np.where(self.env.grid == CellType.WALL.value)
        return list(zip(wall_locations[1], wall_locations[0]))

    def get_empty_cells(self):
        empty_cells = np.where(self.env.grid == CellType.OPEN.value)
        return list(zip(empty_cells[1], empty_cells[0]))

    def set_walls(self, walls):
        # First, clear all existing walls
        wall_locations = np.where(self.env.grid == CellType.WALL.value)
        for y, x in zip(wall_locations[0], wall_locations[1]):
            self.env.grid[y][x] = CellType.OPEN.value

        # Then, add the new walls
        for x, y in walls:
            self.add_wall(x, y)

    def get_num_walls(self):
        return len(self.get_wall_locations())

    def set_start(self, x: int, y: int) -> None:
        self.env.startPositions.append((y, x))
        self.env.grid[y][x] = CellType.START.value

    def set_finish(self, x: int, y: int) -> None:
        self.env.goalPositions.append((y, x))
        self.env.grid[y][x] = CellType.GOAL.value

    def add_rock(self, x: int, y: int) -> None:
        self.env.grid[y][x] = CellType.ROCK.value

    def add_wall(self, x: int, y: int) -> None:
        self.env.step((y, x))

    def remove_wall(self, x: int, y: int) -> None:
        self.env.grid[y][x] = CellType.OPEN.value

    def add_checkpoint(self, x: int, y: int, label: str) -> None:
        checkpoint_index = ord(label) - ord('A')
        self.env.checkpoints.append((y, x, checkpoint_index))
        self.env.grid[y][x] = self.env._checkpointIndexToCellValue(checkpoint_index)


    def find_path(self):
        if not self.env.checkpoints:
            path = self.env._calculateShortestPathFromMultipleStarts(self.env.startPositions, self.env.goalPositions[0])
            return (path, len(path)) if path is not None and len(path) > 0 else (None, 0)

        checkpoints_by_label = {}
        for y, x, label_index in self.env.checkpoints:
            label = chr(ord('A') + label_index)
            if label not in checkpoints_by_label:
                checkpoints_by_label[label] = []
            checkpoints_by_label[label].append((y,x))

        sorted_labels = sorted(checkpoints_by_label.keys())

        total_path = []
        current_pos = self.env.startPositions[0]

        for label in sorted_labels:
            next_checkpoint = None
            shortest_path_to_checkpoint = None

            for checkpoint_pos in checkpoints_by_label[label]:
                path_segment = self.env._calculateShortestSubpath(current_pos, checkpoint_pos)
                if path_segment and (shortest_path_to_checkpoint is None or len(path_segment) < len(shortest_path_to_checkpoint)):
                    shortest_path_to_checkpoint = path_segment
                    next_checkpoint = checkpoint_pos
            
            if not shortest_path_to_checkpoint:
                return None, 0 # No path to the next checkpoint
            
            total_path.extend(shortest_path_to_checkpoint[:-1]) # Avoid duplicating the checkpoint itself
            current_pos = next_checkpoint

        # Path from the last checkpoint to the finish
        final_segment = self.env._calculateShortestSubpath(current_pos, self.env.goalPositions[0])
        if not final_segment:
            return None, 0 # No path from the last checkpoint to the finish

        total_path.extend(final_segment)
        return total_path, len(total_path)

    def draw_path(self, path) -> None:
        # The PatheryEnv does not support drawing the path directly.
        # We will ignore this for now.
        pass

    def display(self) -> None:
        print(self.env.render())
