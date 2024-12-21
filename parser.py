import argparse

from models.graph import Graph
from models.matrix import Matrix
from models.node import Node
from models.edge import Edge, EdgeDirection

def parse_args():
    parser = argparse.ArgumentParser(description="Script that performs Graph Theory operations")

    parser.add_argument('input', type=str, help="Input file with Graph data")

    return parser.parse_args()

def load_graph(args):
    graph = Graph()

    with open(args.input, 'r') as file:
        content = file.read()

        # ; is end of the line
        lines = content.split(';')

        for line in lines:
            line = line.strip().split(' ')

            if 'u' in line[0]:
                if len(line) < 1:
                    raise ValueError('Line does not contain any node info: {}'.format(line))
                
                node = Node(line[1])

                if len(line) > 2:
                    print(line)
                    node.weight = line[2]

                graph.add_node(node=node)

            if 'h' in line[0]:
                if len(line) < 1:
                    raise ValueError('Line does not contain necessary edge info: {}'.format(line))
                
                edge = Edge(start_node=Node(line[1]), direction=EdgeDirection.from_string(line[2]), end_node=Node(line[3]))

                if len(line) >= 5:
                    if line[4].startswith(':'):
                        edge.name = line[4].strip(':')
                    else:
                        edge.weight = int(line[4])

                if len(line) >= 6:
                    edge.name = line[5].strip(':')

                graph.add_edge(edge=edge)

        return graph
    
def read_and_return_input(valid_inputs) -> int:
    option = int(input())
    if option not in valid_inputs:
        print('Not a valid option, ending script...')
        return -1
    return option
    
def print_options_and_return() -> int:
    valid_inputs = [1, 2, 3, 4, 5, 6]
    print('{}. Properties'.format(valid_inputs[0]))
    print('{}. Characteristics'.format(valid_inputs[1]))
    print('{}. Matrix'.format(valid_inputs[2]))
    print('{}. Trace'.format(valid_inputs[3]))
    print('{}. Print graph'.format(valid_inputs[4]))
    print('{}. BFS & DFS'.format(valid_inputs[5]))
    
    option = read_and_return_input(valid_inputs)
    return option

def read_node_name(graph: Graph) -> str:
    if len(graph.sorted_nodes) == 0:
        graph.sort_nodes()

    print('Pick node name:')
    print(graph.sorted_nodes)
    node_name = str(input()).capitalize()
    if node_name not in graph.sorted_nodes:
        print('Not a valid option, ending script...')
        return -1
    return node_name

def read_edge_name(graph: Graph) -> str:
    print('Pick edge name:')
    print(graph.edge_names)
    edge_name = str(input())
    if edge_name not in graph.edge_names:
        print('Not a valid option, ending script...')
        return -1
    return edge_name

def print_matrix_options_and_return() -> int:
    valid_inputs = [1, 2, 3, 4, 5]
    print('{}. Adjacency matrix (Matice sousednosti)'.format(valid_inputs[0]))
    print('{}. Incidence matrix (Matice incidence)'.format(valid_inputs[1]))
    print('{}. Length matrix (Matice delek)'.format(valid_inputs[2]))
    print('{}. Predecessor matrix (Matice predchudcu)'.format(valid_inputs[3]))
    print('{}. Laplace matrix (Laplacelova matice)'.format(valid_inputs[4]))

    option = read_and_return_input(valid_inputs)
    return option

def print_matrix_operations_and_return() -> int:
    valid_inputs = [1, 2, 3]
    print('{}. Get specific value'.format(valid_inputs[0]))
    print('{}. Get num of x'.format(valid_inputs[1]))
    print('{}. Print full matrix'.format(valid_inputs[2]))

    option = read_and_return_input(valid_inputs)
    return option

def print_trace_options_and_return() -> int:
    valid_inputs = [1, 2]
    print('{}. Trace matrix (sled matice)'.format(valid_inputs[0]))
    print('{}. Specific node trace (sled uzlu)'.format(valid_inputs[1]))

    option = read_and_return_input(valid_inputs)
    return option

def print_bfs_dfs_options_and_return() -> int:
    valid_inputs = [1, 2]
    print('{}. Get index'.format(valid_inputs[0]))
    print('{}. Print'.format(valid_inputs[1]))

    option = read_and_return_input(valid_inputs)
    return option

def read_and_return_power() -> int:
    print('Enter power (n)')
    return int(input())

def number_of_x_in_matrix(matrix: Matrix):
    print('Number of what?')
    num = input()

    try:
        num = int(num)
    except ValueError:
        pass

    print('Whole matrix: {}'.format(matrix.get_number_of_x(num)))
    print('Diag: {}'.format(matrix.get_number_of_x_on_diag(num)))

def handle_characteristics(graph: Graph):
    graph.get_ready_for_characteristics()
    node_name = read_node_name(graph)
    print()

    graph.print_characteristics(node_name)

def handle_matrix(graph: Graph):
    graph.get_ready_for_matrix_operations()
    option = print_matrix_options_and_return()
    print()

    # Matice sousednosti
    if option == 1:
        option = print_matrix_operations_and_return()
        print()

        if option == 1:
            print('Enter node 1:')
            node_1 = read_node_name(graph)
            print('Enter node 2:')
            node_2 = read_node_name(graph)
            print(graph.get_specific_adj_point(node_1, node_2))

        elif option == 2:
            number_of_x_in_matrix(graph.adjacency_matrix())

        elif option == 3:
            graph.print_adjacency_matrix()

        elif option == -1:
            pass
    
    # Matice incidence
    elif option == 2:
        option = print_matrix_operations_and_return()
        print()

        if option == 1:
            print('Enter node:')
            node = read_node_name(graph)
            print('Enter edge:')
            edge = read_edge_name(graph)
            print(graph.get_specific_incidence_point(node, edge))

        elif option == 2:
            number_of_x_in_matrix(graph.incidence_matrix())

        elif option == 3:
            graph.print_incidence_matrix()

        elif option == -1:
            pass
    
    # Matice delek
    elif option == 3:
        option = print_matrix_operations_and_return()
        print()

        if option == 1:
            print('Enter node 1:')
            node_1 = read_node_name(graph)
            print('Enter node 2:')
            node_2 = read_node_name(graph)
            print(graph.get_specific_length_point(node_1, node_2))

        elif option == 2:
            number_of_x_in_matrix(graph.length_matrix())

        elif option == 3:
            graph.print_length_matrix()

        elif option == -1:
            pass
    
    # Matice predchudcu
    elif option == 4:
        option = print_matrix_operations_and_return()
        print()

        if option == 1:
            print('Enter node 1:')
            node_1 = read_node_name(graph)
            print('Enter node 2:')
            node_2 = read_node_name(graph)
            print(graph.get_specific_predecessor_point(node_1, node_2))

        elif option == 2:
            number_of_x_in_matrix(graph.predecessor_matrix())

        elif option == 3:
            graph.print_predecessor_matrix()

        elif option == -1:
            pass

    # Laplacelova matice
    elif option == 5:
        pass
    
    elif option == -1:
        pass

def handle_trace(graph: Graph):
    graph.get_ready_for_matrix_operations()
    option = print_trace_options_and_return()
    print()
    
    # Trace matrix
    if option == 1:
        power = read_and_return_power()
        print()
        graph.print_trace_matrix(power)

    # Specific trace
    elif option == 2:
        power = read_and_return_power()
        print('From:')
        node_1 = read_node_name(graph)
        print('To:')
        node_2 = read_node_name(graph)

        print(graph.get_specific_trace(power, node_1, node_2))

def handle_bfs_and_dfs(graph: Graph):
    start_node_name = read_node_name(graph)
    option = print_bfs_dfs_options_and_return()
    graph.get_ready_for_characteristics()

    bfs = graph.bfs(start_node_name)
    dfs = graph.dfs(start_node_name)
    graph.save_bfs_dfs_to_txt(bfs, dfs)

    if option == 1:
        print('Enter index:')
        index = int(input())

        print('Index: {}'.format(index))
        print('BFS (do sirky): {}'.format(bfs[index]))
        print('DFS (do hloubky): {}'.format(dfs[index]))

    elif option == 2:
        print('BFS (do sirky): {}'.format(bfs))
        print('DFS (do hloubky): {}'.format(dfs))

    else:
        pass

def main():
    args = parse_args()
    main_graph = Graph()

    try:
        main_graph = load_graph(args)
        print('Num of nodes: {}'.format(main_graph.get_number_of_nodes()))
        print('Num of edges: {}'.format(main_graph.get_number_of_edges()))
        option = print_options_and_return()
        print()

        # PROPERTIES
        if option == 1:
            main_graph.print_properties()
        
        # CHARACTERISTICS
        elif option == 2:
            handle_characteristics(graph=main_graph)

        # MATRIX
        elif option == 3:
            handle_matrix(main_graph)
        
        # TRACE
        elif option == 4:
            handle_trace(main_graph)
        
        # PRINT WHOLE GRAPH
        elif option == 5:
            main_graph.print_graph()

        # BFS & DFS
        elif option == 6:
            handle_bfs_and_dfs(main_graph)

        elif option == -1:
            pass

    except FileNotFoundError:
        print('Error: Input file with name {} does not exist.'.format(args.input))
    except ValueError as e:
        print('Error:', e)

if __name__ == '__main__':
    main()