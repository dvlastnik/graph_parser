from typing import Set, List
from models.node import Node
from models.edge import Edge

class Graph:
    def __init__(self):
        self.nodes: Set[Node] = set()
        self.edges: List[Edge] = []
        self.weighted = False
        self.two_way = False
    
    # UTILITIES
    def add_node(self, node: Node):
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        if edge in self.edges and edge.name != '':
            raise ValueError(f'Edge with name: {edge.name} is already in list of edges.')

        self.edges.append(edge)

        if edge.weight > 1:
            self.weighted = True

    def print_graph(self):
        for edge in self.edges:
            print(edge.to_string())

    # PROPERTIES METHODS
    def is_weighted(self):
        return self.weighted
    
    def is_two_way(self):
        return self.two_way