class Node:
    def __init__(self, name: str, value=0):
        self.name = name
        self.value = value

    def to_string(self):
        if self.value == 0:
            return f'Node({self.name})'
        return f'Node({self.name}, {self.value})'
