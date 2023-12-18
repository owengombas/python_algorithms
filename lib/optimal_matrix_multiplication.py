from typing import List, Generator, Tuple
import math


def find_best_splitting_points(
    dimensions: List[int], start: int = None, end: int = None
) -> Tuple[int, List[int], List[int], int, int]:
    """
    Finds the best pivot to use in the matrix multiplication

    Complexity:
        - Assume len(dimensions) = n
        - Time: O(n^3) (without memoization: O(n^2), related to cantalan number)
        - Space: O(n^2)

    Args:
        dimensions: list of dimensions of matrices
        start: start index
        end: end index
    """
    if start is None:
        start = 0
    if end is None:
        end = len(dimensions) - 1

    best_value = [[0 for _ in range(end + 1)] for _ in range(end + 1)]
    best_pivot = [[0 for _ in range(end + 1)] for _ in range(end + 1)]
    h = 2

    while h <= end:
        l = h - 2
        while l >= start:
            min_value = float("inf")
            i = l + 1
            while i <= h - 1:
                value = (
                    dimensions[l] * dimensions[i] * dimensions[h]
                    + best_value[l][i]
                    + best_value[i][h]
                )
                if min_value > value:
                    min_value = value
                    best_pivot[l][h] = i
                i += 1
            best_value[l][h] = min_value
            l -= 1
        h += 1

    return best_value[start][end], best_pivot, best_value, start, end


def get_splitting_points(
    pivots: List[List[int]], start: int, end: int
) -> Generator[Tuple[int, int], None, None]:
    """
    Processes the pivots to generate the order of multiplication

    Complexity:
        - Assume len(pivots) = n
        - Time: O(n)
        - Space: O(1)

    Args:
        pivots: list of pivots
        start: start index
        end: end index
    """
    if start + 1 < end:
        b = pivots[start][end]
        yield from get_splitting_points(pivots, start, b)
        yield from get_splitting_points(pivots, b, end)
        yield b


def cantalan(n: int) -> int:
    """
    Calculates the cantalan number

    Complexity:
        - Assume n is the number of matrices
        - Time: O(n)
        - Space: O(1)

    Args:
        n: number of matrices
    """
    return int((1 / (n + 1)) * math.comb(2 * n, n))
