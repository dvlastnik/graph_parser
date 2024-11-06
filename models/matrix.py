from typing import List

class Matrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0] * cols for _ in range(rows)]

    def set_value(self, row: int, col: int, value: int):
        self.matrix[row][col] = value

    def get_value(self, row: int, col: int):
        return self.matrix[row][col]
    
    def increment_value(self, row: int, col: int):
        self.matrix[row][col] += 1
    
    def print_matrix(self):
        for row in self.matrix:
            print(row)

    def print_matrix_with_headers(self, row_headers: List[str] = None, col_headers: List[str] = None):
        if col_headers:
            print(' ', ' '.join(col_headers))
        for i, row in enumerate(self.matrix):
            row_str = ''

            for r in row:
                row_str += str(r)
                if r >= 0:
                    row_str += ' ' * len(col_headers[0])
                elif r < 0:
                    row_str += ' ' * (len(col_headers[0]) - 1)

            row_name = row_headers[i] if row_headers else str(i)
            print(f"{row_name} {row_str}")
        print()