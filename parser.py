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

                if len(line) >= 5:
                    edge.weight = int(line[4])

                if len(line) >= 6:
                    edge.name = line[5].strip(':')

                graph.add_edge(edge=edge)

        graph.print_graph()

def main():
    args = parse_args()

    try:
        load_graph(args)
    except FileNotFoundError:
        print(f'Error: Input file with name {args.input} does not exist.')
    except ValueError as e:
        print('Error:', e)

if __name__ == '__main__':
    main()