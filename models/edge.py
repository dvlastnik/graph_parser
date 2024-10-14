from enum import Enum

from models.node import Node

class EdgeDirection(Enum):
    FORWARD = '>'
    REVERSE = '<'
    BOTH = '-'

    @staticmethod
    def from_string(value: str):
        return EdgeDirection(value=value)
    
class Edge:
    def __init__(self, start_node: Node, direction: EdgeDirection, end_node: Node):
        self.start_node = start_node
        self.direction = direction
        self.end_node = end_node
        self.weight = 1
        self.name = ''

    def __eq__(self, edge):
        if isinstance(edge, Edge):
            return self.name == edge.name
        return False

    def to_string(self):
        result = f'{self.start_node.to_string()}'

        if self.direction == EdgeDirection.REVERSE:
            result += f'{self.direction.value}-'
        else:
            result += '--'

        if self.weight == 0:
            result += f'--'
        else:
            result += f'{self.name}({self.weight})'

        if self.direction == EdgeDirection.FORWARD:
            result += f'-{self.direction.value}'
        else:
            result += '--'

        result += f'{self.end_node.to_string()}'

        return result
        