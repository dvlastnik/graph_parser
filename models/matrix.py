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

    def fill_matrix(self, value):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = value

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

    def determinant(self) -> int:
        if self.rows != self.cols:
            raise ValueError("Determinant can only be computed for square matrices.")

        if self.rows == 1:
            return self.matrix[0][0]

        if self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]

        if self.rows == 3:
            return (
                self.matrix[0][0] * self.matrix[1][1] * self.matrix[2][2]
                + self.matrix[0][1] * self.matrix[1][2] * self.matrix[2][0]
                + self.matrix[0][2] * self.matrix[1][0] * self.matrix[2][1]
                - self.matrix[0][2] * self.matrix[1][1] * self.matrix[2][0]
                - self.matrix[0][0] * self.matrix[1][2] * self.matrix[2][1]
                - self.matrix[0][1] * self.matrix[1][0] * self.matrix[2][2]
            )

        # Recursive case for larger matrices using Leibniz formula
        det = 0
        for col in range(self.cols):
            # Get the submatrix excluding the first row and the current column
            submatrix = self._get_submatrix(0, col)

            # Add or subtract the submatrix determinant based on column index
            det += ((-1) ** col) * self.matrix[0][col] * submatrix.determinant()

        return det
    
    def _get_submatrix(self, exclude_row: int, exclude_col: int) -> 'Matrix':
        submatrix = Matrix(self.rows - 1, self.cols - 1)
        for i in range(self.rows):
            if i == exclude_row:
                continue
            for j in range(self.cols):
                if j == exclude_col:
                    continue
                new_row = i - 1 if i > exclude_row else i
                new_col = j - 1 if j > exclude_col else j
                submatrix.set_value(new_row, new_col, self.matrix[i][j])
        return submatrix
    
    def delete_last_row_col(self):
        if self.rows <= 1 or self.cols <= 1:
            raise ValueError("Cannot delete last row and column from a matrix smaller than 2x2.")

        new_matrix = [
            [self.matrix[i][j] for j in range(self.cols - 1)]
            for i in range(self.rows - 1)
        ]

        self.matrix = new_matrix
        self.rows -= 1
        self.cols -= 1