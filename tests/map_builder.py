
class MapBuilder:
    """
    A helper class to build map strings for PatheryEnv tests.
    """
    def __init__(self, width: int, height: int, num_walls: int = 0):
        self.width = width
        self.height = height
        self.num_walls = num_walls
        self.objects = []

    def set_start(self, x: int, y: int):
        self.objects.append({'x': x, 'y': y, 'type': 's1'})
        return self

    def set_finish(self, x: int, y: int):
        self.objects.append({'x': x, 'y': y, 'type': 'f1'})
        return self

    def add_rock(self, x: int, y: int):
        self.objects.append({'x': x, 'y': y, 'type': 'r1'})
        return self

    def add_checkpoint(self, x: int, y: int, label: str):
        checkpoint_num = ord(label) - ord('A') + 1
        self.objects.append({'x': x, 'y': y, 'type': f'c{checkpoint_num}'})
        return self

    def build(self) -> str:
        """
        Builds the map string from the configured objects.
        Format: <width>.<height>.<num_walls>:<sparse_map>
        """
        # Add a dummy name part to the metadata
        metadata = f"{self.width}.{self.height}.{self.num_walls}.test"

        if not self.objects:
            return f"{metadata}:"

        # Sort objects by their position in a row-major traversal
        self.objects.sort(key=lambda obj: obj['y'] * self.width + obj['x'])

        map_parts = []
        last_linear_pos = -1

        for obj in self.objects:
            linear_pos = obj['y'] * self.width + obj['x']
            free_cells = linear_pos - last_linear_pos - 1
            map_parts.append(f"{free_cells},{obj['type']}")
            last_linear_pos = linear_pos

        return f"{metadata}:" + ".".join(map_parts)
