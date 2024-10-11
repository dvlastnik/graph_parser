class Node:
    def __init__(self, name: str):
        self.name = name
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)