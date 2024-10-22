import argparse

from models.graph import Graph
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
                    raise ValueError(f'Line does not contain any node info: {line}')
                
                node = Node(line[1])

                if len(line) > 2:
                    print(line)
                    node.weight = line[2]

                graph.add_node(node=node)

            if 'h' in line[0]:
                if len(line) < 1:
                    raise ValueError(f'Line does not contain necessary edge info: {line}')
                
                edge = Edge(start_node=Node(line[1]), direction=EdgeDirection.from_string(line[2]), end_node=Node(line[3]))
                if EdgeDirection.from_string(line[2]) == EdgeDirection.REVERSE:
                    edge = Edge(start_node=Node(line[3]), direction=EdgeDirection.FORWARD, end_node=Node(line[1]))

                if len(line) >= 5:
                    if line[4].startswith(':'):
                        edge.name = line[4].strip(':')
                    else:
                        edge.weight = int(line[4])

                if len(line) >= 6:
                    edge.name = line[5].strip(':')

                graph.add_edge(edge=edge)

        return graph
    
def print_options_and_return() -> int:
    valid_inputs = [1, 2, 3]
    print(f'{valid_inputs[0]}. Properties')
    print(f'{valid_inputs[1]}. Matrix')
    print(f'{valid_inputs[2]}. Print graph')
    
    option = int(input())
    if option not in valid_inputs:
        print('Not a valid option, ending script...')
        return -1
    return option

def main():
    args = parse_args()
    main_graph = Graph()

    try:
        main_graph = load_graph(args)
        
        option = print_options_and_return()
        if option == 1:
            main_graph.print_properties()
        elif option == 2:
            main_graph.print_adjacency_matrix()
        elif option == 3:
            main_graph.print_graph()
        elif option == -1:
            pass

    except FileNotFoundError:
        print(f'Error: Input file with name {args.input} does not exist.')
    except ValueError as e:
        print('Error:', e)

if __name__ == '__main__':
    main()