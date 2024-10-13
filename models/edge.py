from node import Node

class Edge:
    def __init__(self, name, start_node: Node, end_node: Node, weight=1,):
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight

    def __eq__(self, edge):
        if isinstance(edge, Edge):
            return self.name == edge.name
        return False
        

    def to_string(self):
        result = f'Edge({self.start_node.to_string()}--'

        if self.weight == 0:
            result += f'{self.name}({self.weight})'
        else:
            result += f'--'

        result += f'--'
        result += f'>{self.end_node.to_string()}'
        result += f')'

        return result
        