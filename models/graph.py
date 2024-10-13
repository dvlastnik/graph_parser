from node import Node
from edge import Edge

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = list()
        self.weighted = False
    
    # UTILITIES
    def add_node(self, node: Node):
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        if edge in self.edges:
            raise ValueError(f'Edge with name: {edge.name} is already in list of edges.')

        self.edges.append(edge)

        if edge.weight > 1:
            self.weighted = True

    # PROPERTIES METHODS
    def is_weighted(self):
        return self.weighted