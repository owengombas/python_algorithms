import math
import time
from typing import List
from lib.matrix_exponent import MatrixExponent


class Fibonacci:
    """
    Generates the Fibonacci sequence up to the nth number.
    Args:
        n (int): The length of the Fibonacci sequence to generate.
    Returns:
        list: The Fibonacci sequence up to the nth number.

    Usage example:
        ```python
        fibonacci = Fibonacci()
        fibonacci.recursive(10) # This will take a long time and it returns a list of the first 10 Fibonacci numbers.
        fibonacci.iterative(10) # This returns a list of the first 10 Fibonacci numbers.

        fibonacci.matrix(10) # This returns the 10th Fibonacci number.
        fibonacci.closed_form(10) # This returns the 10th Fibonacci number.
        ```
    """

    def recursive(self, n: int) -> List[int]:
        """
        Generates the Fibonacci sequence up to the nth number using recursion.

        Complexity:
            Time: O(2^n)
            Space: O(n)
        Args:
            n (int): The length of the Fibonacci sequence to generate.
        Returns:
            list: The Fibonacci sequence up to the nth number.
        """
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]

        fib_seq = self.recursive(n - 1)
        fib_seq.append(fib_seq[-1] + fib_seq[-2])

        return fib_seq

    def iterative(self, n: int) -> List[int]:
        """
        Generates the Fibonacci sequence up to the nth number using iteration.

        Complexity:
            Time: O(n)
            Space: O(n)
        Args:
            n (int): The length of the Fibonacci sequence to generate.
        Returns:
            list: The Fibonacci sequence up to the nth number.
        """
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]

        fib_seq = [0, 1]
        for _ in range(2, n):
            fib_seq.append(fib_seq[-1] + fib_seq[-2])

        return fib_seq

    def matrix(self, n: int) -> int:
        """
        Generates the nth Fibonacci number using matrix exponentiation.
        The matrix exponentiation method is defined as:
        F_n = [[1, 1], [1, 0]]^(n-1)[0][0]
        It's slower than the closed form, however it's more precise for large n.

        Complexity:
            Time: O(log(n))
            Space: O(1)
        Args:
            n (int): The index of the Fibonacci number to generate.
        Returns:
            int: The nth Fibonacci number.
        """
        if n <= 0:
            return 0

        return MatrixExponent().strassen([[1, 1], [1, 0]], n - 1)[0][0]

    def closed_form(self, n: int) -> int:
        """
        Generates the nth Fibonacci number using the closed form.
        The closed form is defined as:
        F_n = ((1 + sqrt(5))^n - (1 - sqrt(5))^n) / (2^n * sqrt(5))
        It's the fastest method, but it's not very precise for large n.

        Complexity:
            Time: O(1)
            Space: O(1)
        Args:
            n (int): The index of the Fibonacci number to generate.
        Returns:
            int: The nth Fibonacci number.
        """
        sqrt_5 = math.sqrt(5)
        return int(((1 + sqrt_5) ** n - (1 - sqrt_5) ** n) / (2**n * sqrt_5))


if __name__ == "__main__":
    fibonacci = Fibonacci()
    n = 20

    start = time.time()
    expected = fibonacci.recursive(n)
    end = time.time()
    print(f"Recursive: {expected}")
    print(f"Time: {end - start}")

    start = time.time()
    actual = fibonacci.iterative(n)
    end = time.time()
    print(f"Iterative: {actual}")
    print(f"Time: {end - start}")
    assert len(actual) == n, f"{len(actual)} != {n}"
    assert actual == expected, f"{actual} != {expected}"

    start = time.time()
    actual = fibonacci.closed_form(n - 1)
    end = time.time()
    print(f"Closed form: {actual}")
    print(f"Time: {end - start}")
    assert actual == expected[-1], f"{actual} != {expected[-1]}"

    start = time.time()
    actual = fibonacci.matrix(n - 1)
    end = time.time()
    print(f"Matrix: {actual}")
    print(f"Time: {end - start}")
    assert actual == expected[-1], f"{actual} != {expected[-1]}"
