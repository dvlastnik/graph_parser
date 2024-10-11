class Edge:
    def __init__(self, name, start_node, end_node, weight=0, is_two_way=False):
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.is_two_way = is_two_way

    def to_string(self):
        result = f'Edge({self.start_node}--'

        if self.weight == 0:
            result += f'{self.weight}'
        else:
            result += f'--'

        result += f'--'

        if self.is_two_way:
            result += f'{self.end_node}'
        else:
            result += f'>{self.end_node}'

        result += f')'

        return result
        