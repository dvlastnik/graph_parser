class Node:
    def __init__(self, name: str, weight=0):
        self.name = name
        self.weight = weight

    def __lt__(self, other):
        return self.name < other.name
    
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name and self.weight == other.weight
        return False

    def to_string(self):
        if self.weight == 0:
            return '({})'.format(self.name)
        return '({}, {})'.format(self.name, self.weight)
