from typing import Set, List
from collections import deque
from itertools import combinations

from models.node import Node
from models.edge import Edge, EdgeDirection

class Graph:
    def __init__(self):
        self.nodes: Set[Node] = set()
        self.edges: List[Edge] = []

        self.edges_set = set()
        self.nodes_map = {}

        self.weighted = False
        self.directed = False
        self.simple = False
    
    # UTILITIES
    def add_node(self, node: Node):
        self.nodes.add(node)
        self.nodes_map[node.name] = 0

    def add_edge(self, edge: Edge):
        if edge in self.edges and edge.name != '':
            raise ValueError(f'Edge with name: {edge.name} is already in list of edges.')

        self.edges.append(edge)
        self.nodes_map[edge.start_node.name] += 1
        self.nodes_map[edge.end_node.name] += 1

        node_pair = None
        if edge.direction == EdgeDirection.BOTH:
            self.directed = False
            node_pair = frozenset([edge.start_node, edge.end_node])
        else:
            self.directed = True
            node_pair = (edge.start_node, edge.end_node)

        if edge.weight > 1:
            self.weighted = True
        
        if node_pair in self.edges_set:
            self.simple = False
        else:
            self.edges_set.add(node_pair)
            self.simple = True

    def contains_k5(self):
        if len(self.nodes) < 5:
            return False
        
        for comb in combinations(self.nodes, 5):
            subgraph_edges = [(u, v) for u, v in combinations(comb, 2)]
            if all(edge in self.edges or (edge[1], edge[0]) in self.edges for edge in subgraph_edges):
                return True
        return False

    def contains_k33(self):
        if len(self.nodes) < 6:
            return False
        
        for left_part in combinations(self.nodes, 3):
            right_part = self.nodes - set(left_part)
            if len(right_part) < 3:
                continue
            right_part = list(right_part)[:3]
            
            subgraph_edges = [(u, v) for u in left_part for v in right_part]
            if all(edge in self.edges or (edge[1], edge[0]) in self.edges for edge in subgraph_edges):
                return True
        return False

    def print_graph(self):
        for edge in self.edges:
            print(edge.to_string())

    # PROPERTIES METHODS
    # Prazdny
    def is_empty(self) -> bool:
        return len(self.edges) == 0 and len(self.nodes) == 0
    
    # Diskretni
    def is_discrate(self) -> bool:
        if len(self.nodes) > 0 and len(self.edges) == 0:
            return True
            
        return False

    # Prosty
    def is_simple(self) -> bool:
        return self.simple
    
    # Jednoduchy
    def is_easy(self) -> bool:
        if self.simple:
            for edge in self.edges:
                if edge.start_node == edge.end_node:
                    return False # loop detected
            return True
        return False

    # Multigraf
    def is_multi(self) -> bool:
        return not self.is_simple()

    # Orientovany
    def is_directed(self) -> bool:
        if self.directed:
            for edge in self.edges:
                reversed_edge = Edge(start_node=edge.end_node, end_node=edge.start_node, direction=edge.direction)

                if reversed_edge not in self.edges:
                    return True
                
        return False

    # Ohodnoceny
    def is_weighted(self) -> bool:
        return self.weighted
    
    # Uplny
    def is_complete(self) -> bool:
        num_nodes = len(self.nodes)
        if num_nodes <= 1:
            return True

        expected_edges = 0
        if self.directed:
            expected_edges = num_nodes * (num_nodes - 1)
        else:
            expected_edges = (num_nodes * (num_nodes - 1)) / 2

        if len(self.edges_set) == expected_edges:
            return True
        return False

    # Bipartitni
    def is_bipartite(self) -> bool: 
        if not self.nodes:
            return True

        color = {}

        # BFS function to check bipartiteness from a given node
        def bfs_check(start_node):
            queue = deque([start_node])
            color[start_node] = 0

            while queue:
                node = queue.popleft()
                current_color = color[node]

                for edge in self.edges:
                    if edge.start_node == node:
                        neighbor = edge.end_node
                    elif edge.end_node == node:
                        neighbor = edge.start_node
                    else:
                        continue

                    if neighbor not in color:
                        color[neighbor] = 1 - current_color
                        queue.append(neighbor)
                    elif color[neighbor] == current_color:
                        return False

            return True
        
        for node in self.nodes:
            if node not in color:
                if not bfs_check(node):
                    return False

        return True

    # Souvisly
    def is_connected(self) -> bool:
        num_nodes = len(self.nodes)
        if num_nodes <= 1:
            return True

        visited_nodes = set()
        start_node = next(iter(self.nodes))

        queue = deque([start_node])
        visited_nodes.add(start_node)

        while queue:
            node = queue.popleft()

            for edge in self.edges:
                if edge.start_node == node and edge.end_node not in visited_nodes:
                    visited_nodes.add(edge.end_node)
                    queue.append(edge.end_node)
                elif edge.end_node == node and edge.start_node not in visited_nodes:
                    visited_nodes.add(edge.start_node)
                    queue.append(edge.start_node)

        return len(visited_nodes) == len(self.nodes)

    # Rovinny
    def is_planar(self):
        return not (self.contains_k5() or self.contains_k33())

    # Konecny
    def is_finite(self):
        return len(self.nodes) < float('inf') and len(self.edges) < float('inf')

    # Regularni    
    def is_regular(self):
        values = iter(self.nodes_map.values())
        first_value = next(values)

        if all(value == first_value for value in values):
            return True
        else:
            return False

    def print_properties(self):
        print(f'Empty (prazdny): {self.is_empty()}')
        print(f'Discrate (diskretni): {self.is_discrate()}')
        print(f'Simple (prosty): {self.is_simple()}')
        print(f'Easy (jednoduchy): {self.is_easy()}')
        print(f'Multi (multigraf): {self.is_multi()}')
        print(f'Directed (orientovany): {self.is_directed()}')
        print(f'Weighted (ohodnoceny): {self.is_weighted()}')
        print(f'Complete (uplny): {self.is_complete()}')
        print(f'Bipartite (bipartitni): {self.is_bipartite()}')
        print(f'Connected (souvisly): {self.is_connected()}')
        print(f'Planar (rovinny): {self.is_planar()}')
        print(f'Finite (konecny): {self.is_finite()}')
        print(f'Regular (regularni): {self.is_regular()}')