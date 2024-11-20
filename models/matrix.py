class Matrix:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0] * cols for _ in range(rows)]

    def set_value(self, row: int, col: int, value):
        self.matrix[row][col] = value

    def get_value(self, row: int, col: int):
        return self.matrix[row][col]
    
    def increment_value(self, row: int, col: int):
        self.matrix[row][col] += 1

    def get_number_of_x(self, num: int):
        result = 0

        for row in self.matrix:
            for col in row:
                if col == num:
                    result += 1

        return result

    def get_number_of_x_on_diag(self, num: int) -> int:
        result = 0

        for i, row in enumerate(self.matrix):
            if self.matrix[i][i] == num:
                result += 1
        
        return result

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
    
    def multiply_self_on_n(self, n: int) -> 'Matrix':
        if n < 2:
            return self

        result = self

        for _ in range(n - 1):
            result = result.multiply(self)

        return result
    
    def print_matrix(self):
        for row in self.matrix:
            print(row)

    # type: (list[str], list[str])
    def print_matrix_with_headers(self, row_headers=None, col_headers=None):
        if col_headers:
            print(' ', ' '.join(col_headers))
        for i, row in enumerate(self.matrix):
            row_str = ''
 
            for j, r in enumerate(row):
                row_str += str(r)

                if isinstance(r, (int, float)):
                    if r >= 0:
                        row_str += ' ' * len(col_headers[j])
                    elif r < 0:
                        row_str += ' ' * (len(col_headers[j]) - 1)
                else:
                    if len(r) >= 0:
                        row_str += ' ' * len(col_headers[j])
                    elif len(r) < 0:
                        row_str += ' ' * (len(col_headers[j]) - 1)

            row_name = row_headers[i] if row_headers else str(i)
            print('{} {}'.format(row_name, row_str))
        print()

    def save_matrix_with_headers(self, filename, row_headers=None, col_headers=None):
        with open(filename, 'w') as file:
            if col_headers:
                file.write('  ' + ' '.join(col_headers) + '\n')
            
            for i, row in enumerate(self.matrix):
                row_str = ''
                
                for j, r in enumerate(row):
                    row_str += str(r)

                    if isinstance(r, (int, float)):
                        if r >= 0:
                            row_str += ' ' * len(col_headers[j])
                        elif r < 0:
                            row_str += ' ' * (len(col_headers[j]) - 1)
                    else:
                        if len(r) >= 0:
                            row_str += ' ' * len(col_headers[j])
                        elif len(r) < 0:
                            row_str += ' ' * (len(col_headers[j]) - 1)
                
                row_name = row_headers[i] if row_headers else str(i)
                file.write('{} {}\n'.format(row_name, row_str))