from enum import Enum
import re

from models.node import Node

class EdgeDirection(Enum):
    FORWARD = '>'
    REVERSE = '<'
    BOTH = '-'

    @staticmethod
    def from_string(value: str):
        return EdgeDirection(value=value)
    
    @staticmethod
    def get_opposite(value):
        if value == EdgeDirection.FORWARD:
            return EdgeDirection.REVERSE
        return EdgeDirection.FORWARD
    
class Edge:
    def __init__(self, start_node: Node, direction: EdgeDirection, end_node: Node, weight=1, name=''):
        self.start_node = start_node
        self.direction = direction
        self.end_node = end_node
        self.weight = weight
        self.name = name

    def __eq__(self, edge):
        if isinstance(edge, Edge):
            return self.start_node == edge.start_node and self.end_node == edge.end_node and self.direction == edge.direction
        return False
    
    @staticmethod
    def extract_number_from_name(edge):
        result = re.search(r'\d+$', edge.name)

        return int(result.group()) if result else float('inf')

    def to_string(self):
        result = '{}'.format(self.start_node.to_string())

        if self.direction == EdgeDirection.REVERSE:
            result += '{}-'.format(self.direction.value)
        else:
            result += '--'

        if self.weight == 0:
            result += '--'
        else:
            result += '{}({})'.format(self.name, self.weight)

        if self.direction == EdgeDirection.FORWARD:
            result += '-{}'.format(self.direction.value)
        else:
            result += '--'

        result += '{}'.format(self.end_node.to_string())

        return result
        