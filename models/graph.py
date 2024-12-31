from collections import deque
import heapq
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

    def get_node_by_name(self, node_name: str) -> Node:
        for node in self.nodes:
            if node.name == node_name:
                return node
            
        return None

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
                normalized_edge = Edge(start_node=edge.end_node, direction=EdgeDirection.get_opposite(edge.direction), end_node=edge.start_node, name=edge.name, weight=edge.weight)
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
    
    def get_node_output_edges(self, node_name: str) -> list:
        result = []

        for edge in self.normalized_edges:
            if self.directed:
                if edge.start_node.name == node_name:
                    result.append(edge)
            else:
                if edge.start_node.name == node_name:
                    result.append(edge)
                elif edge.end_node.name == node_name:
                    result.append(edge)

        return result

    def get_node_output_neighborhood(self, node_name: str) -> set:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.start_node.name == node_name:
                    if edge.name != '':
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
            else:
                if edge.start_node.name == node_name:
                    if edge.name != '':
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
                elif edge.end_node.name == node_name:
                    if edge.name != '':
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)

        return result

    def get_node_input_neighborhood(self, node_name: str) -> set:
        result = set()

        for edge in self.normalized_edges:
            if self.directed:
                if edge.end_node.name == node_name:
                    if edge.name != '':
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
            else:
                if edge.start_node.name == node_name:
                    if edge.name != '':
                        result.add(edge.name)
                    else:
                        result.add(edge.weight)
                elif edge.end_node.name == node_name:
                    if edge.name != '':
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

    def _print_marix(self, matrix: Matrix, cols: list, rows: list, filename: str):
        matrix.print_matrix_with_headers(rows, cols)
        matrix.save_matrix_with_headers(filename, rows, cols)
    
    def _get_specific_point(self, matrix: Matrix, node_1: str, node_2: str):
        node_1_index = 0
        node_2_index = 0

        for i, node in enumerate(self.sorted_nodes):
            if node == node_1:
                node_1_index = i
            if node == node_2:
                node_2_index = i

        return matrix.get_value(node_1_index, node_2_index)

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
        return self._get_specific_point(self.adjacency_matrix(), node_1, node_2)
    
    def print_adjacency_matrix(self):
        self._print_marix(self.adjacency_matrix(), self.sorted_nodes, self.sorted_nodes, 'adjacency_matrix.txt')

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
    
    # Matice delek
    def length_matrix(self) -> Matrix:
        matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.sorted_nodes))
        node_index = {node: i for i, node in enumerate(self.sorted_nodes)}

        for edge in self.normalized_edges:
            start = edge.start_node
            end = edge.end_node
            start_idx = node_index[start.name]
            end_idx = node_index[end.name]

            matrix.set_value(start_idx, end_idx, edge.weight)

            if not self.directed:
                matrix.set_value(end_idx, start_idx, edge.weight)

        for i, rows in enumerate(matrix.matrix):
            for j, col in enumerate(rows):
                if col == 0 and i != j:
                    matrix.set_value(i, j, '∞')

        return matrix
    
    def print_length_matrix(self):
        self._print_marix(self.length_matrix(), self.sorted_nodes, self.sorted_nodes, 'length_matrix.txt')

    def get_specific_length_point(self, node_1: str, node_2: str) -> str:
        return self._get_specific_point(self.length_matrix(), node_1, node_2)
    
    # Matice predchudcu
    def predecessor_matrix(self) -> Matrix:
        matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.sorted_nodes))
        matrix.fill_matrix('-')
        node_index = {node: i for i, node in enumerate(self.sorted_nodes)}

        for edge in self.normalized_edges:
            start = edge.start_node
            end = edge.end_node
            start_idx = node_index[start.name]
            end_idx = node_index[end.name]

            matrix.set_value(start_idx, end_idx, start.name)

            if not self.directed:
                matrix.set_value(end_idx, start_idx, end.name)

        return matrix

    def print_predecessor_matrix(self):
        self._print_marix(self.predecessor_matrix(), self.sorted_nodes, self.sorted_nodes, 'predecessor_matrix.txt')

    def get_specific_predecessor_point(self, node_1: str, node_2: str) -> str:
        return self._get_specific_point(self.predecessor_matrix(), node_1, node_2)
    
    # Matice stupnu
    def degree_matrix(self) -> Matrix:
        matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.sorted_nodes))
        matrix.fill_matrix(0)

        for i, node_name in enumerate(self.sorted_nodes):
            matrix.set_value(i, i, len(self.get_node_neighbors(node_name)))

        return matrix
    
    def print_degree_matrix(self):
        self._print_marix(self.degree_matrix(), self.sorted_nodes, self.sorted_nodes, 'degree_matrix.txt')

    def get_specific_degree_point(self, node_1: str, node_2: str) -> str:
        return self._get_specific_point(self.degree_matrix(), node_1, node_2)
    
    # Laplacelova matice
    def laplace_matrix(self) -> Matrix:
        adjacency_matrix = self.adjacency_matrix()
        degree_matrix = self.degree_matrix()

        laplace_matrix = Matrix(rows=len(self.sorted_nodes), cols=len(self.sorted_nodes))
        for i in range(len(self.sorted_nodes)):
            for j in range(len(self.sorted_nodes)):
                if i == j:
                    # Diagonal element: Degree
                    laplace_matrix.set_value(i, j, degree_matrix.get_value(i, j))
                else:
                    # Off-diagonal element: -Adjacency
                    laplace_matrix.set_value(i, j, -adjacency_matrix.get_value(i, j))

        return laplace_matrix

    def print_laplace_matrix(self):
        self._print_marix(self.laplace_matrix(), self.sorted_nodes, self.sorted_nodes, 'laplace_matrix.txt')

    def get_specific_laplace_point(self, node_1: str, node_2: str):
        return self._get_specific_point(self.laplace_matrix(), node_1, node_2)

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

    # BFS & DFS
    # BFS
    def bfs(self, start_node_name: str) -> list:
        visited = set()
        result = []
        queue = deque([start_node_name])
        visited.add(start_node_name)

        while queue:
            node_name = queue.popleft()
            #print('node name {}'.format(node_name))
            result.append(node_name)
            #print('result {}'.format(result))

            for successor in sorted(self.get_node_successors(node_name)):
                if successor not in visited:
                    visited.add(successor)
                    queue.append(successor)

        return result

    # DFS
    def dfs(self, start_node_name: str, visited=None, result=[], successors=None):
        if visited is None:
            visited = set()

        visited.add(start_node_name)
        result.append(start_node_name)

        if successors is None:
            for successor in sorted(self.get_node_successors(start_node_name)):
                if successor not in visited:
                    result = self.dfs(successor, visited, result)
        else:
            for successor in successors:
                if successor not in visited:
                    result = self.dfs(successor, visited, result)

        return result
    
    def save_bfs_dfs_to_txt(self, bfs: list, dfs: list):
        try:
            with open('bfs.txt', 'w') as file:
                for node_name in bfs:
                    file.write('{} '.format(node_name))

            with open('dfs.txt', 'w') as file:
                for node_name in dfs:
                    file.write('{} '.format(node_name))

        except Exception as e:
            print("Error occurred while saving the results: {}".format(e))

    # BONES OF GRAPH
    def get_number_of_bones(self) -> int:
        self.get_ready_for_matrix_operations()
        matrix = self.laplace_matrix()
        matrix.delete_last_row_col()

        return matrix.determinant()
    
    # Minimalni kostra
    def find(self, parent, node):
        if parent[node] != node:
            parent[node] = self.find(parent, parent[node])
        return parent[node]

    def union(self, parent, rank, u, v):
        root_u = self.find(parent, u)
        root_v = self.find(parent, v)

        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    def kruskal_minumum(self) -> tuple:
        parent = {node.name: node.name for node in self.nodes}
        rank = {node.name: 0 for node in self.nodes}

        sorted_edges = sorted(self.normalized_edges, key=lambda e: e.weight)

        mst = []
        sum = 0

        for edge in sorted_edges:
            start = edge.start_node.name
            end = edge.end_node.name

            if self.find(parent, start) != self.find(parent, end):
                mst.append(edge.name)
                self.union(parent, rank, start, end)
                sum += edge.weight

        return mst, sum
    
    def kruskal_maximum(self) -> tuple:
        parent = {node.name: node.name for node in self.nodes}
        rank = {node.name: 0 for node in self.nodes}

        sorted_edges = sorted(self.normalized_edges, key=lambda e: e.weight, reverse=True)

        mst = []
        sum = 0

        for edge in sorted_edges:
            start = edge.start_node.name
            end = edge.end_node.name

            if self.find(parent, start) != self.find(parent, end):
                mst.append(edge.name)
                sum += edge.weight
                self.union(parent, rank, start, end)

        return mst, sum
    
    # BINARY TREE
    def is_binary_tree(self) -> bool:
        is_directed = self.is_directed()

        if not self.is_connected():
            return False

        if self.get_number_of_edges() != self.get_number_of_nodes() - 1:
            return False

        for node in self.sorted_nodes:
            degree = 0
            if is_directed:
                degree = self.get_node_stage(node)
            else:
                degree = self.get_node_input_stage(node)

            if degree > 3:
                return False

        return True
    
    def preorder(self, node_name: str, visited: set):
        if node_name is None or node_name in visited:
            return []

        visited.add(node_name)

        successors = sorted(self.get_node_successors(node_name))
        traversal = [node_name]
        for successor in successors:
            traversal += self.preorder(successor, visited)
        return traversal
    
    def inorder(self, node_name: str, visited: set, parent_name: str = None):
        if node_name is None or node_name in visited:
            return []

        visited.add(node_name)

        successors = [succ for succ in self.get_node_successors(node_name) if succ != parent_name]

        left = self.inorder(successors[0], visited, node_name) if len(successors) > 0 else []
        right = self.inorder(successors[1], visited, node_name) if len(successors) > 1 else []

        return left + [node_name] + right
    
    def postorder(self, node_name: str, visited: set):
        if node_name is None or node_name in visited:
            return []

        visited.add(node_name)

        successors = sorted(self.get_node_successors(node_name))
        traversal = []
        for successor in successors:
            traversal += self.postorder(successor, visited)
        return traversal + [node_name]
    
    def get_root_node(self) -> Node:
        in_degree = {node.name: 0 for node in self.nodes}

        for edge in self.edges:
            in_degree[edge.end_node.name] += 1

        for node in self.nodes:
            if in_degree[node.name] == 0:
                return node

        return None
    
    def print_binary_tree_chars(self):
        self.get_ready_for_characteristics()
        if self.is_binary_tree():
            print('This graph is binary tree')
            root = self.get_root_node()

            preorder_result = ''
            inorder_result = ''
            postorder_result = ''

            preorder = self.preorder(root.name, set())
            inorder = self.inorder(root.name, set())
            postorder = self.postorder(root.name, set())

            if len(preorder) != len(inorder) or len(inorder) != len(postorder) or len(preorder) != len(postorder):
                raise Exception('Length of orders are not same: Preorder {}, inorder {}, postorder {}'.format(len(preorder), len(inorder), len(postorder)))

            for i in range(len(preorder)):
                preorder_result += preorder[i]
                inorder_result += inorder[i]
                postorder_result += postorder[i]

            # Print the results
            print("Preorder: {}".format(preorder_result))
            print("Inorder: {}".format(inorder_result))
            print("Postorder: {}".format(postorder_result))
        else:
            print('This graph is NOT binary tree')

    # SHORTEST PATH
    def can_use_moore(self) -> bool:
        if not self.is_weighted():
            return True
        
        for edge in self.edges:
            if edge.weight != 1:
                return False
        return True
    
    def can_use_dijkstra(self) -> bool:
        if not self.is_weighted():
            return False
        
        for edge in self.edges:
            if edge.weight < 0:
                return False
        return True
    
    def can_use_bellman_ford(self) -> bool:
        return self.is_weighted()
    
    def can_use_floyd_warshall(self):
        if not self.is_weighted():
            return False

        if self.has_negative_cycle():
            return False

        return True

    # Checking negative cycles, floyd marshall cant be used on graph with negative cycle
    def has_negative_cycle(self):
        distances = {node.name: float('inf') for node in self.nodes}
        start_node = next(iter(self.nodes))
        distances[start_node.name] = 0

        for _ in range(len(self.nodes) - 1):
            for edge in self.edges:
                if distances[edge.start_node.name] + edge.weight < distances[edge.end_node.name]:
                    distances[edge.end_node.name] = distances[edge.start_node.name] + edge.weight

        for edge in self.edges:
            if distances[edge.start_node.name] + edge.weight < distances[edge.end_node.name]:
                return True

        return False
    
    def save_shortest_path_to_file(self, name: str, results: dict):
        try:
            with open(name, 'w') as file:
                for node in results.keys():
                    r = '{} = ('.format(node)

                    for i in range(len(results[node])):
                        if i == len(results[node])-1:
                            r += '{})\n'.format(results[node][i])
                        else:
                            r += '{}, '.format(results[node][i])
                    
                    file.write(r)

        except Exception as e:
            print("Error occurred while saving the results: {}".format(e))
    
    def print_properties_for_shortest_path(self, distances: dict, filename: str):
        if float('inf') in distances.values():
            print('Float(inf) is in this shortest path')

        reachable_distances = [dist[1] for dist in distances.values() if dist[1] != float('inf')]
        max_length = max(reachable_distances, default=None)
        min_length = min(reachable_distances, default=None)
        not_connected_count = sum(1 for dist in distances.values() if dist[1] == float('inf'))
        connected_count = len(reachable_distances)

        print('Maximal path length: {}'.format(max_length))
        print('Minimal path length: {}'.format(min_length))
        print('Number of nodes not connected: {}'.format(not_connected_count))
        print('Number of nodes connected: {}'.format(connected_count))
        
        self.save_shortest_path_to_file(filename, distances)

    def get_path_from_distances(self, distances: dict, start_node: str, end_node: str) -> list:
        if distances[end_node][1] == float('inf'):
            return f"No path exists from {start_node} to {end_node}."

        path = []
        current_node = end_node

        while current_node != '-':
            path.append(current_node)
            current_node = distances[current_node][0]

        path.reverse()
        path_str = ''.join(path)
        length = distances[end_node][1]

        return path_str, length
    
    def get_path_from_floyd(self, predecessor_matrix: Matrix, start_node: str, end_node: str):
        start_idx = self.sorted_nodes.index(start_node)
        end_idx = self.sorted_nodes.index(end_node)

        if predecessor_matrix.get_value(start_idx, end_idx) is None:
            return f"No path exists from {start_node} to {end_node}."

        path = [end_node]
        while start_idx != end_idx:
            end_idx = predecessor_matrix.get_value(start_idx, end_idx)
            path.append(self.sorted_nodes[end_idx])

        path.reverse()
        path_str = ''.join(path)
        return path_str
    
    def moore_shortest_path(self, start_node_name: str):
        distances = {node.name: ['-', float('inf')] for node in self.nodes}
        distances[start_node_name][1] = 0
        queue = deque([start_node_name])

        while queue:
            current_node = queue.popleft()

            for edge in self.normalized_edges:
                if edge.start_node.name == current_node:
                    neighbor = edge.end_node.name
                    if distances[neighbor][1] == float('inf'):
                        distances[neighbor][1] = distances[current_node][1] + 1
                        distances[neighbor][0] = current_node
                        queue.append(neighbor)

        self.print_properties_for_shortest_path(distances=distances, filename='moore.txt')

        return distances
    
    def dijkstra_shortest_path(self, start_node_name: str):
        distances = {node.name: ['-', float('inf')] for node in self.nodes}
        distances[start_node_name][1] = 0
        priority_queue = [(0, start_node_name)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node][1]:
                continue

            for edge in self.normalized_edges:
                if edge.start_node.name == current_node:
                    neighbor = edge.end_node.name
                    new_distance = current_distance + edge.weight

                    if new_distance < distances[neighbor][1]:
                        distances[neighbor][1] = new_distance
                        distances[neighbor][0] = current_node
                        heapq.heappush(priority_queue, (new_distance, neighbor))

        self.print_properties_for_shortest_path(distances=distances, filename='dijkstra.txt')
        return distances
    
    def bellman_ford_shortest_path(self, start_node_name: str):
        results = {node.name: ['-', float('inf'), 0] for node in self.nodes}
        results[start_node_name][1] = 0

        queue = deque([start_node_name])

        while queue:
            current_node_name = queue.popleft()

            edges = self.get_node_output_edges(current_node_name)
            for edge in edges:
                if self.directed:
                    start = edge.start_node.name
                    end = edge.end_node.name
                    weight = edge.weight
                    if results[start][1] + weight < results[end][1]:
                        results[end][1] = results[start][1] + weight
                        results[end][0] = start

                        results[end][2] = results[start][2] + 1

                        queue.append(end)
                else:
                    start = edge.start_node.name
                    end = edge.end_node.name
                    weight = edge.weight

                    if results[start][1] + weight < results[end][1]:
                        results[end][1] = results[start][1] + weight
                        results[end][0] = start

                        results[end][2] = results[start][2] + 1

                        queue.append(end)
                    elif results[end][1] + weight < results[start][1]:
                        results[start][1] = results[end][1] + weight
                        results[start][0] = end

                        results[start][2] = results[end][2] + 1

                        queue.append(start)

        self.print_properties_for_shortest_path(distances=results, filename='dijkstra.txt')
        return results
    
    def floyd_warshall(self):
        dist_matrix = self.length_matrix()
        num_nodes = dist_matrix.rows

        predecessor_matrix = Matrix(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                predecessor_matrix.set_value(i, j, i if dist_matrix.get_value(i, j) not in [float('inf'), '∞'] else None)

        for i in range(num_nodes):
            for j in range(num_nodes):
                value = dist_matrix.get_value(i, j)
                if value == '∞' or not isinstance(value, (int, float)):
                    dist_matrix.set_value(i, j, float('inf'))

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    current_distance = dist_matrix.get_value(i, j)
                    through_k_distance = dist_matrix.get_value(i, k) + dist_matrix.get_value(k, j)

                    if through_k_distance < current_distance:
                        dist_matrix.set_value(i, j, through_k_distance)
                        predecessor_matrix.set_value(i, j, predecessor_matrix.get_value(k, j))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist_matrix.get_value(i, j) == float('inf'):
                    dist_matrix.set_value(i, j, '∞')

        reachable_distances = []
        not_connected_count = 0

        for i in range(num_nodes):
            for j in range(num_nodes):
                value = dist_matrix.get_value(i, j)
                if value != '∞' and i != j:
                    reachable_distances.append(value)
                elif value == '∞':
                    not_connected_count += 1

        max_length = max(reachable_distances, default=None)
        min_length = min(reachable_distances, default=None)
        connected_count = len(reachable_distances)

        print(f"Maximal path length: {max_length}")
        print(f"Minimal path length: {min_length}")
        print(f"Number of connected paths: {connected_count}")
        print(f"Number of unconnected paths: {not_connected_count}")

        dist_matrix.save_matrix_with_headers('floyd_warshall.txt', self.sorted_nodes, self.sorted_nodes)
        predecessor_matrix.save_matrix_with_headers('floyd_warshall_.txt', self.sorted_nodes, self.sorted_nodes)
        dist_matrix.print_matrix_with_headers(self.sorted_nodes, self.sorted_nodes)
        
        return dist_matrix, predecessor_matrix