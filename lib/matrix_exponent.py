from typing import List
from lib.matrix_multiplication import MatrixMultiplication


class MatrixExponent:
    """
    Computes the exponent of a matrix.
    Args:
        A (list): The matrix.
        n (int): The exponent.
    Returns:
        list: The exponent of the matrix.
    Usage example:
        ```python
        A = [[1, 2], [3, 4]]
        matrix_exponent = MatrixExponent()
        matrix_exponent.naive(A, 3)
        matrix_exponent.strassen(A, 3)
        ```
    """

    def naive(self, A: List[List[int]], n: int) -> List[List[int]]:
        """
        Computes the exponent of a matrix using the naive algorithm.
        Complexity:
            Time: O(n^3)
            Space: O(n^2)
        Args:
            A (list): The matrix.
            n (int): The exponent.
        Returns:
            list: The exponent of the matrix.
        """
        if len(A) != len(A[0]):
            raise ValueError("A must be a square matrix.")

        if n == 0:
            return [[1, 0], [0, 1]]

        if n == 1:
            return A

        return MatrixMultiplication().naive(A, self.naive(A, n - 1))

    def strassen(self, A: List[List[int]], n: int) -> List[List[int]]:
        """
        Computes the exponent of a matrix using the Strassen algorithm.
        Complexity:
            Time: O(n^2.81)
            Space: O(n^2)
        Args:
            A (list): The matrix.
            n (int): The exponent.
        Returns:
            list: The exponent of the matrix.
        """
        if len(A) != len(A[0]):
            raise ValueError("A must be a square matrix.")

        if n == 0:
            return [[1, 0], [0, 1]]

        if n == 1:
            return A

        return MatrixMultiplication().strassen(
            A, self.strassen(A, n - 1)
        )


if __name__ == "__main__":
    matrix_exponent = MatrixExponent()
    M1 = [[1, 2], [3, 4]]
    M2 = [[7, 8], [9, 10], [11, 12]]
    naive = matrix_exponent.naive(M1, 3)
    strassen = matrix_exponent.strassen(M1, 3)
    expected = [[37, 54], [81, 118]]
    assert naive == expected, f"{naive} != {expected}"
    assert strassen == expected, f"{strassen} != {expected}"
