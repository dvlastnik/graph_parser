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
    valid_inputs = [1, 2, 3, 4, 5]
    print('{}. Properties'.format(valid_inputs[0]))
    print('{}. Characteristics'.format(valid_inputs[1]))
    print('{}. Matrix'.format(valid_inputs[2]))
    print('{}. Trace'.format(valid_inputs[3]))
    print('{}. Print graph'.format(valid_inputs[4]))
    
    option = read_and_return_input(valid_inputs)
    return option

def read_node_name(graph: Graph) -> str:
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
            main_graph.get_ready_for_characteristics()
            node_name = read_node_name(main_graph)
            print()

            main_graph.print_characteristics(node_name)

        # MATRIX
        elif option == 3:
            main_graph.get_ready_for_matrix_operations()
            option = print_matrix_options_and_return()
            print()

            # Matice sousednosti
            if option == 1:
                option = print_matrix_operations_and_return()
                print()

                if option == 1:
                    print('Enter node 1:')
                    node_1 = read_node_name(main_graph)
                    print('Enter node 2:')
                    node_2 = read_node_name(main_graph)
                    print(main_graph.get_specific_adj_point(node_1, node_2))

                elif option == 2:
                    number_of_x_in_matrix(main_graph.adjacency_matrix())

                elif option == 3:
                    main_graph.print_adjacency_matrix()

                elif option == -1:
                    pass
            
            # Matice incidence
            elif option == 2:
                option = print_matrix_operations_and_return()
                print()

                if option == 1:
                    print('Enter node:')
                    node = read_node_name(main_graph)
                    print('Enter edge:')
                    edge = read_edge_name(main_graph)
                    print(main_graph.get_specific_incidence_point(node, edge))

                elif option == 2:
                    number_of_x_in_matrix(main_graph.incidence_matrix())

                elif option == 3:
                    main_graph.print_incidence_matrix()

                elif option == -1:
                    pass
            
            # Matice delek
            elif option == 3:
                option = print_matrix_operations_and_return()
                print()

                if option == 1:
                    print('Enter node 1:')
                    node_1 = read_node_name(main_graph)
                    print('Enter node 2:')
                    node_2 = read_node_name(main_graph)
                    print(main_graph.get_specific_length_point(node_1, node_2))

                elif option == 2:
                    number_of_x_in_matrix(main_graph.length_matrix())

                elif option == 3:
                    main_graph.print_length_matrix()

                elif option == -1:
                    pass
            
            # Matice predchudcu
            elif option == 4:
                pass

            # Laplacelova matice
            elif option == 5:
                pass
            
            elif option == -1:
                pass
        
        # TRACE
        elif option == 4:
            main_graph.get_ready_for_matrix_operations()
            option = print_trace_options_and_return()
            print()
            
            # Trace matrix
            if option == 1:
                power = read_and_return_power()
                print()
                main_graph.print_trace_matrix(power)

            # Specific trace
            elif option == 2:
                power = read_and_return_power()
                print('From:')
                node_1 = read_node_name(main_graph)
                print('To:')
                node_2 = read_node_name(main_graph)

                print(main_graph.get_specific_trace(power, node_1, node_2))
        
        # PRINT WHOLE GRAPH
        elif option == 5:
            main_graph.print_graph()

        elif option == -1:
            pass

    except FileNotFoundError:
        print('Error: Input file with name {} does not exist.'.format(args.input))
    except ValueError as e:
        print('Error:', e)

if __name__ == '__main__':
    main()