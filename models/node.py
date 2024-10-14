class Node:
    def __init__(self, name: str, weight=0):
        self.name = name
        self.weight = weight

    def to_string(self):
        if self.weight == 0:
            return f'({self.name})'
        return f'({self.name}, {self.weight})'
