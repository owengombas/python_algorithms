from typing import List


class MatrixMultiplication:
    """
    Computes the product of two matrices.
    Args:
        A (list): The first matrix.
        B (list): The second matrix.
    Returns:
        list: The product of the two matrices.
    Usage example:
        ```python
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        matrix_multiplication = MatrixMultiplication()
        matrix_multiplication.naive(A, B)
        matrix_multiplication.strassen(A, B)
        ```
    """

    def naive(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Computes the product of two matrices using the naive algorithm.
        Complexity:
            Time: O(n^3)
            Space: O(n^2)
        Args:
            A (list): The first matrix.
            B (list): The second matrix.
        Returns:
            list: The product of the two matrices.
        """
        if len(A[0]) != len(B):
            raise ValueError(
                "The number of columns in A must match the number of rows in B."
            )

        n = len(A)
        m = len(B[0])
        C = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                for k in range(len(B)):
                    C[i][j] += A[i][k] * B[k][j]

        return C
    
    def add(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Adds two matrices.
        Args:
            A (list): The first matrix.
            B (list): The second matrix.
        Returns:
            list: The sum of the two matrices.
        """
        n = len(A)
        m = len(A[0])
        C = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                C[i][j] = A[i][j] + B[i][j]

        return C
    
    def subtract(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Subtracts two matrices.
        Args:
            A (list): The first matrix.
            B (list): The second matrix.
        Returns:
            list: The difference of the two matrices.
        """
        n = len(A)
        m = len(A[0])
        C = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                C[i][j] = A[i][j] - B[i][j]

        return C
    
    def split(self, A: List[List[int]]) -> List[List[int]]:
        """
        Splits a matrix into four submatrices.
        Args:
            A (list): The matrix.
        Returns:
            list: The four submatrices.
        """
        n = len(A) // 2
        A11 = [[0 for _ in range(n)] for _ in range(n)]
        A12 = [[0 for _ in range(n)] for _ in range(n)]
        A21 = [[0 for _ in range(n)] for _ in range(n)]
        A22 = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                A11[i][j] = A[i][j]
                A12[i][j] = A[i][j + n]
                A21[i][j] = A[i + n][j]
                A22[i][j] = A[i + n][j + n]

        return A11, A12, A21, A22
    
    def merge(
        self,
        A11: List[List[int]],
        A12: List[List[int]],
        A21: List[List[int]],
        A22: List[List[int]],
    ) -> List[List[int]]:
        """
        Merges four submatrices into a matrix.
        Args:
            A11 (list): The first submatrix.
            A12 (list): The second submatrix.
            A21 (list): The third submatrix.
            A22 (list): The fourth submatrix.
        Returns:
            list: The merged matrix.
        """
        n = len(A11)
        A = [[0 for _ in range(2 * n)] for _ in range(2 * n)]

        for i in range(n):
            for j in range(n):
                A[i][j] = A11[i][j]
                A[i][j + n] = A12[i][j]
                A[i + n][j] = A21[i][j]
                A[i + n][j + n] = A22[i][j]

        return A
    
    def strassen(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        """
        Computes the product of two matrices using the Strassen algorithm.
        Complexity:
            Time: O(n^2.81)
            Space: O(n^2)
        Args:
            A (list): The first matrix.
            B (list): The second matrix.
        Returns:
            list: The product of the two matrices.
        """
        if len(A[0]) != len(B):
            raise ValueError(
                "The number of columns in A must match the number of rows in B."
            )

        n = len(A)
        m = len(B[0])
        C = [[0 for _ in range(m)] for _ in range(n)]

        if n == 1:
            C[0][0] = A[0][0] * B[0][0]
        else:
            A11, A12, A21, A22 = self.split(A)
            B11, B12, B21, B22 = self.split(B)

            M1 = self.strassen(self.add(A11, A22), self.add(B11, B22))
            M2 = self.strassen(self.add(A21, A22), B11)
            M3 = self.strassen(A11, self.subtract(B12, B22))
            M4 = self.strassen(A22, self.subtract(B21, B11))
            M5 = self.strassen(self.add(A11, A12), B22)
            M6 = self.strassen(self.subtract(A21, A11), self.add(B11, B12))
            M7 = self.strassen(self.subtract(A12, A22), self.add(B21, B22))

            C11 = self.add(self.subtract(self.add(M1, M4), M5), M7)
            C12 = self.add(M3, M5)
            C21 = self.add(M2, M4)
            C22 = self.add(self.add(self.subtract(M1, M2), M3), M6)

            C = self.merge(C11, C12, C21, C22)

        return C


if __name__ == "__main__":
    M1 = [[1, 2], [3, 4]]
    M2 = [[5, 6], [7, 8]]

    matrix_multiplication = MatrixMultiplication()

    naive = matrix_multiplication.naive(M1, M2)
    strassen = matrix_multiplication.strassen(M1, M2)
    expected = [[19, 22], [43, 50]]
    assert naive == expected, f"{naive} != {expected}"
    assert strassen == expected, f"{strassen} != {expected}"

