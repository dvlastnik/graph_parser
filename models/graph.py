from collections import deque
from itertools import combinations

from models.matrix import Matrix
from models.node import Node
from models.edge import Edge, EdgeDirection

class Graph:
    def __init__(self):
        self.nodes = set()  # type: set[Node]
        self.edges = []  # type: list[Edge]

        self.edges_set = set()
        self.sorted_nodes = []  # type: list[str]
        self.nodes_map = {}
        self.normalized_edges = []  # type: list[Edge]
        self.edge_names = []

        self.weighted = False
        self.directed = False
        self.simple = True
    
    # UTILITIES
    def add_node(self, node: Node):
        self.nodes.add(node)
        self.nodes_map[node.name] = 0

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self.nodes_map[edge.start_node.name] += 1
        self.nodes_map[edge.end_node.name] += 1

        if edge.direction == EdgeDirection.BOTH:
            self.directed = False
        else:
            self.directed = True

        if edge.weight > 1:
            self.weighted = True

    def get_number_of_nodes(self) -> int:
        return len(self.nodes)
    
    def get_number_of_edges(self) -> int:
        return len(self.edges)

    def sort_nodes(self):
        for node in self.nodes:
            self.sorted_nodes.append(node.name)
        
        self.sorted_nodes = sorted(self.sorted_nodes)

    def sort_edges(self):
        self.edges.sort(key=Edge.extract_number_from_name)
        
        for edge in self.edges:
            self.edge_names.append(edge.name)

    def normalize_edges(self):
        for edge in self.edges:
            if edge.direction == EdgeDirection.REVERSE:
                normalized_edge = Edge(start_node=edge.end_node, direction=EdgeDirection.get_opposite(edge.direction), end_node=edge.start_node, name=edge.name)
                self.normalized_edges.append(normalized_edge)
            else:
                self.normalized_edges.append(edge)
    
    def get_edge_names(self):
        return [edge.name for edge in self.normalized_edges]

    def print_graph(self):
        for edge in self.edges:
            print(edge.to_string())

    def print_normalized_graph(self):
        for edge in self.normalized_edges:
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
        self.normalize_edges()
        node_pair = None

        for edge in self.normalized_edges:
            if edge.direction == EdgeDirection.BOTH:
                self.directed = False
                node_pair = frozenset([edge.start_node, edge.end_node])
            else:
                self.directed = True
                node_pair = (edge.start_node, edge.end_node)
            
            if node_pair in self.edges_set:
                self.simple = False
            else:
                self.edges_set.add(node_pair)

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
        return not self.simple

    # Orientovany
    def is_directed(self) -> bool:
        return self.directed

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
        if self.is_weighted():
            print('Ohodnoceny (weighted)')
        if self.is_directed():
            print('Orientovany (directed)')
        if self.is_empty():
            print('Prazdny (empty)')
        if self.is_simple():
            print('Prosty (simple)')
        if self.is_easy():
            print('Jednoduchy (easy)')
        if self.is_discrate():
            print('Diskretni (discrate)')
        if self.is_multi():
            print('Multigraf (multigraph)')
        if self.is_complete():
            print('Uplny (complete)')
        if self.is_bipartite():
            print('Bipartitni (bipartite)')
        if self.is_connected():
            print('Souvisly (connected)')
        if self.is_finite():
            print('Konecny (finite)')
        if self.is_regular():
            print('Regularni (regular)')

    # CHARACTERISTICS
    def get_ready_for_characteristics(self):
        self.sort_nodes()
        self.sort_edges()
        self.normalize_edges()

    def get_node_successors(self, node_name: str) -> list:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.start_node.name == node_name:
                    result.add(edge.end_node.name)
            else:
                if edge.start_node.name == node_name:
                    result.add(edge.end_node.name)
                elif edge.end_node.name == node_name:
                    result.add(edge.start_node.name)

        return result
    
    def get_node_ancestors(self, node_name: str) -> set:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.end_node.name == node_name:
                    result.add(edge.start_node.name)
            else:
                if edge.start_node.name == node_name:
                    result.add(edge.end_node.name)
                elif edge.end_node.name == node_name:
                    result.add(edge.start_node.name)

        return result
    
    def get_node_neighbors(self, node_name: str) -> set:
        return self.get_node_ancestors(node_name) | self.get_node_successors(node_name)
    
    def get_node_output_neighborhood(self, node_name: str) -> set:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.start_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
            else:
                if edge.start_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
                elif edge.end_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)

        return result

    def get_node_input_neighborhood(self, node_name: str) -> set:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.end_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
            else:
                if edge.start_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
                elif edge.end_node.name == node_name:
                    if edge.name is not None:
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)

        return result

    def get_node_neighborhood(self, node_name: str) -> set:
        return self.get_node_output_neighborhood(node_name) | self.get_node_input_neighborhood(node_name)

    def get_node_output_stage(self, node_name: str) -> int:
        result = 0

        for edge in self.normalized_edges:
            if self.directed:
                if edge.start_node.name == node_name:
                    result += 1
            else:
                if edge.start_node.name == node_name:
                    result += 1
                elif edge.end_node.name == node_name:
                    result += 1

        return result

    def get_node_input_stage(self, node_name: str) -> int:
        result = 0

        for edge in self.normalized_edges:
            if self.directed:
                if edge.end_node.name == node_name:
                    result += 1
            else:
                if edge.start_node.name == node_name:
                    result += 1
                elif edge.end_node.name == node_name:
                    result += 1

        return result

    def get_node_stage(self, node_name: str) -> int:
        return self.get_node_output_stage(node_name) + self.get_node_input_stage(node_name)
    
    def print_characteristics(self, node_name: str):
        print('Ug+({}), nasledniky uzlu: {}'.format(node_name, sorted(self.get_node_successors(node_name))))
        print('Ug-({}), predchudce uzlu: {}'.format(node_name, sorted(self.get_node_ancestors(node_name))))
        print('Ug({}), sousedy uzlu: {}'.format(node_name, sorted(self.get_node_neighbors(node_name))))
        print('Hg+({}), vystupni okoli uzlu: {}'.format(node_name, sorted(self.get_node_output_neighborhood(node_name), key=lambda x: int(x[1:]))))
        print('Hg-({}), vstupni okoli uzlu: {}'.format(node_name, sorted(self.get_node_input_neighborhood(node_name), key=lambda x: int(x[1:]))))
        print('Hg({}), okoli uzlu: {}'.format(node_name, sorted(self.get_node_neighborhood(node_name), key=lambda x: int(x[1:]))))
        print('dg+({}), vystupni stupen uzlu: {}'.format(node_name, self.get_node_output_stage(node_name)))
        print('dg-({}), vstupni stupen uzlu: {}'.format(node_name, self.get_node_input_stage(node_name)))
        print('dg({}), stupen uzlu: {}'.format(node_name, self.get_node_stage(node_name)))

    # MATRIXES
    def get_ready_for_matrix_operations(self):
        self.sort_nodes()
        self.sort_edges()
        self.normalize_edges()

    # Matice sousednosti
    def adjacency_matrix(self) -> Matrix:
        matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.sorted_nodes))
        node_index = {node: i for i, node in enumerate(self.sorted_nodes)}

        for edge in self.normalized_edges:
            start = edge.start_node
            end = edge.end_node
            start_idx = node_index[start.name]
            end_idx = node_index[end.name]

            matrix.increment_value(start_idx, end_idx)

            if not self.directed:
                matrix.increment_value(end_idx, start_idx)

        return matrix
    
    def get_specific_adj_point(self, node_1: str, node_2: str) -> str:
        matrix = self.adjacency_matrix()

        node_1_index = 0
        node_2_index = 0

        for i, node in enumerate(self.sorted_nodes):
            if node == node_1:
                node_1_index = i
            if node == node_2:
                node_2_index = i

        return matrix.get_value(node_1_index, node_2_index)
    
    def print_adjacency_matrix(self):
        matrix = self.adjacency_matrix()
        matrix.print_matrix_with_headers(self.sorted_nodes, self.sorted_nodes)
        matrix.save_matrix_with_headers('adjacency_matrix.txt', self.sorted_nodes, self.sorted_nodes)

    # Matice incidencee
    def incidence_matrix(self) -> Matrix:
        matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.normalized_edges))
        node_index = {node: i for i, node in enumerate(self.sorted_nodes)}

        for col, edge in enumerate(self.normalized_edges):
            start = node_index[edge.start_node.name]
            end = node_index[edge.end_node.name]

            if start == end:
                matrix.set_value(start, col, 2)
            else:
                matrix.set_value(start, col, 1)

                if not self.directed:
                    matrix.set_value(end, col, 1)
                elif self.directed:
                    matrix.set_value(end, col, -1)

        return matrix

    def print_incidence_matrix(self):
        matrix = self.incidence_matrix()
        matrix.print_matrix_with_headers(self.sorted_nodes, self.get_edge_names())
        matrix.save_matrix_with_headers('incidence_matrix.txt', self.sorted_nodes, self.get_edge_names())

    def get_specific_incidence_point(self, node: str, edge: str) -> str:
        matrix = self.incidence_matrix()

        node_index = 0
        edge_index = 0

        for i, node in enumerate(self.sorted_nodes):
            if node == node:
                node_index = i
                break

        for i, edge_name in enumerate(self.edge_names):
            if edge_name == edge:
                edge_index = i
                break

        return matrix.get_value(node_index, edge_index)

    # TRACE
    def trace_matrix(self, power: int) -> Matrix:
        matrix = self.adjacency_matrix()
        matrix = matrix.multiply_self_on_n(power)

        return matrix
    
    def print_trace_matrix(self, power: int):
        matrix = self.trace_matrix(power)
        matrix.print_matrix_with_headers(self.sorted_nodes, self.sorted_nodes)
        matrix.save_matrix_with_headers('trace_matrix.txt', self.sorted_nodes, self.sorted_nodes)

    def get_specific_trace(self, power: int, node_1: str, node_2: str):
        matrix = self.trace_matrix(power)

        node_1_index = 0
        node_2_index = 0

        for i, node in enumerate(self.sorted_nodes):
            if node == node_1:
                node_1_index = i
            if node == node_2:
                node_2_index = i

        return matrix.get_value(node_1_index, node_2_index)