from typing import List


def quicksort(arr: List[float], lower: int, upper: int):
    """
    Sorts a list of numbers using the quicksort algorithm.

    Complexity:
        Time: O(n log n)
        Space: O(log n)

    Args:
        arr: List of numbers to be sorted.
        lower: Lower bound of the list.
        upper: Upper bound of the list.
    """
    if len(arr) == 0:
        return
    if lower < upper:
        i = lower
        j = upper
        k = (lower + upper) // 2
        pivot = arr[k]
        while True:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
            if i > j:
                break
        quicksort(arr, lower, j)
        quicksort(arr, i, upper)
    return arr
