import heapq

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

    def find_path(self):
        """
        Finds the path from the start to the finish using the Pathery AI logic.

        The AI's move priority is: Up, Right, Down, Left.
        """
        if not self.start or not self.finish:
            raise ValueError("Start and finish positions must be set.")

        # Priority queue for the A* algorithm
        # (cost, path)
        pq = [(0, [self.start])]
        visited = set()

        while pq:
            cost, path = heapq.heappop(pq)
            node = path[-1]

            if node == self.finish:
                return path

            if node in visited:
                continue
            visited.add(node)

            x, y = node
            # The order of neighbors is determined by the AI's priority: Up, Right, Down, Left
            neighbors = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]

            for i, (nx, ny) in enumerate(neighbors):
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] not in ('#', 'O'):
                    new_path = list(path)
                    new_path.append((nx, ny))
                    # The cost is the length of the path. We add a small value based on the move priority
                    # to ensure the correct path is chosen when lengths are equal.
                    new_cost = len(new_path) + i * 0.1
                    heapq.heappush(pq, (new_cost, new_path))

        return None  # No path found

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