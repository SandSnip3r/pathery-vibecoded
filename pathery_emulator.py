import heapq
from pathery_pathfinding import find_path_cpp

class PatheryEmulator:
    """
    A text-based emulator for the Pathery puzzle game.
    """

    def __init__(self, width, height, num_walls):
        """
        Initializes the Pathery emulator with a grid of the specified dimensions.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
        """
        self.width = width
        self.height = height
        self.num_walls = num_walls
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.start = None
        self.finish = None
        self.checkpoints = []

    def get_num_walls(self):
        """
        Counts the number of walls currently on the grid.
        """
        return sum(row.count('#') for row in self.grid)

    def set_start(self, x, y):
        """
        Sets the starting position on the grid.

        Args:
            x (int): The x-coordinate of the starting position.
            y (int): The y-coordinate of the starting position.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.start:
                self.grid[self.start[1]][self.start[0]] = ' '
            self.start = (x, y)
            self.grid[y][x] = 'S'
        else:
            raise ValueError("Start position is out of bounds.")

    def set_finish(self, x, y):
        """
        Sets the finishing position on the grid.

        Args:
            x (int): The x-coordinate of the finishing position.
            y (int): The y-coordinate of the finishing position.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.finish:
                self.grid[self.finish[1]][self.finish[0]] = ' '
            self.finish = (x, y)
            self.grid[y][x] = 'F'
        else:
            raise ValueError("Finish position is out of bounds.")

    def add_rock(self, x, y):
        """
        Adds a rock to the grid. Rocks are permanent obstacles.

        Args:
            x (int): The x-coordinate of the rock.
            y (int): The y-coordinate of the rock.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[y][x] not in ('S', 'F'):
                self.grid[y][x] = 'O'
        else:
            raise ValueError("Rock position is out of bounds.")

    def add_wall(self, x, y):
        """
        Adds a wall to the grid.

        Args:
            x (int): The x-coordinate of the wall.
            y (int): The y-coordinate of the wall.
        """
        if self.get_num_walls() >= self.num_walls:
            raise ValueError("Cannot place more walls than the specified limit.")
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[y][x] not in ('S', 'F', 'O'):
                self.grid[y][x] = '#'
        else:
            raise ValueError("Wall position is out of bounds.")

    def remove_wall(self, x, y):
        """
        Removes a wall from the grid.

        Args:
            x (int): The x-coordinate of the wall to remove.
            y (int): The y-coordinate of the wall to remove.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[y][x] == '#':
                self.grid[y][x] = ' '
        else:
            raise ValueError("Wall position is out of bounds.")

    def add_checkpoint(self, x, y, label):
        """
        Adds a checkpoint to the grid.

        Args:
            x (int): The x-coordinate of the checkpoint.
            y (int): The y-coordinate of the checkpoint.
            label (str): The label for the checkpoint (e.g., 'A', 'B').
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[y][x] not in ('S', 'F', 'O', '#'):
                self.grid[y][x] = label
                self.checkpoints.append(((x, y), label))
        else:
            raise ValueError("Checkpoint position is out of bounds.")

    def remove_checkpoint(self, x, y):
        """
        Removes a checkpoint from the grid.

        Args:
            x (int): The x-coordinate of the checkpoint to remove.
            y (int): The y-coordinate of the checkpoint to remove.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.grid[y][x].isalpha() and self.grid[y][x] not in ('S', 'F'):
                self.grid[y][x] = ' '
                self.checkpoints = [(pos, lbl) for pos, lbl in self.checkpoints if pos != (x, y)]
        else:
            raise ValueError("Checkpoint position is out of bounds.")

    def find_path(self):
        """
        Finds the path from the start to the finish, visiting checkpoints in order.
        """
        if not self.start or not self.finish:
            raise ValueError("Start and finish positions must be set.")

        walls = []
        rocks = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == '#':
                    walls.append((x, y))
                elif self.grid[y][x] == 'O':
                    rocks.append((x, y))

        if not self.checkpoints:
            path = find_path_cpp(self.width, self.height, walls, rocks, self.start, self.finish)
            return path if path else None

        # Organize checkpoints by label
        checkpoints_by_label = {}
        for pos, label in self.checkpoints:
            if label not in checkpoints_by_label:
                checkpoints_by_label[label] = []
            checkpoints_by_label[label].append(pos)

        sorted_labels = sorted(checkpoints_by_label.keys())

        total_path = []
        current_pos = self.start

        for label in sorted_labels:
            next_checkpoint = None
            shortest_path_to_checkpoint = None

            for checkpoint_pos in checkpoints_by_label[label]:
                path_segment = find_path_cpp(self.width, self.height, walls, rocks, current_pos, checkpoint_pos)
                if path_segment and (shortest_path_to_checkpoint is None or len(path_segment) < len(shortest_path_to_checkpoint)):
                    shortest_path_to_checkpoint = path_segment
                    next_checkpoint = checkpoint_pos
            
            if not shortest_path_to_checkpoint:
                return None # No path to the next checkpoint
            
            total_path.extend(shortest_path_to_checkpoint[:-1]) # Avoid duplicating the checkpoint itself
            current_pos = next_checkpoint

        # Path from the last checkpoint to the finish
        final_segment = find_path_cpp(self.width, self.height, walls, rocks, current_pos, self.finish)
        if not final_segment:
            return None # No path from the last checkpoint to the finish

        total_path.extend(final_segment)
        return total_path

    def draw_path(self, path):
        """
        Draws the given path on the grid.

        Args:
            path (list): The list of (x, y) tuples representing the path.
        """
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == '.':
                    self.grid[y][x] = ' '

        if path:
            for x, y in path:
                if self.grid[y][x] == ' ':
                    self.grid[y][x] = '.'

    def display(self):
        """
        Displays the current state of the grid with a border.
        """
        # Top border
        print('+' + '-' * (self.width * 2 - 1) + '+')
        for row in self.grid:
            # Side borders
            print('|' + ' '.join(row) + '|')
        # Bottom border
        print('+' + '-' * (self.width * 2 - 1) + '+')

if __name__ == '__main__':
    # Create a 10x10 grid
    game = PatheryEmulator(10, 10, 10)

    # Set start and finish points
    game.set_start(0, 0)
    game.set_finish(9, 9)

    # Add some rocks
    game.add_rock(5, 0)
    game.add_rock(5, 1)
    game.add_rock(5, 2)
    game.add_rock(5, 3)
    game.add_rock(5, 4)

    # Add some walls
    game.add_wall(1, 0)
    game.add_wall(1, 1)
    game.add_wall(1, 2)
    game.add_wall(1, 3)
    game.add_wall(1, 4)


    # Find and display the path
    path = game.find_path()
    if path:
        print(f"Path found with length: {len(path)}")
        game.draw_path(path)
    else:
        print("No path found.")

    # Display the grid
    game.display()