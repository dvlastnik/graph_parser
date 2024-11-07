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

    def multiply(self, other: 'Matrix') -> 'Matrix':
        if self.cols != other.rows:
            raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix.")
        
        result = Matrix(self.rows, other.cols)
        
        for i in range(self.rows):
            for j in range(other.cols):
                sum_product = 0
                for k in range(self.cols):
                    sum_product += self.get_value(i, k) * other.get_value(k, j)
                result.set_value(i, j, sum_product)
        
        return result
    
    def multiply_self_on_n(self, n) -> 'Matrix':
        if n < 2:
            return self

        result = self

        for _ in range(n - 1):
            result.print_matrix()
            print()
            result = result.multiply(self)

        return result
    
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