from typing import List, Tuple, Generator, Generic, TypeVar, Callable, Optional
import graphviz


class Heap:
    @property
    def size(self) -> int:
        return self._size

    @property
    def heap(self) -> List[int]:
        return self._heap

    def __init__(self, array: List[int] = [], fix=True):
        """
        Initialize a heap, if an array is provided, build a heap from the array in O(n) time

        Complexity:
            - Time: O(n)
            - Space: O(n)

        Args:
            array: the array to be converted into a heap
        """
        self._heap = array
        self._size = len(array)
        if fix:
            if self._size > 0:
                for i in range(self._size // 2, -1, -1):
                    self._sift_down(i)

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left(self, i: int) -> int:
        return 2 * i + 1

    def _right(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

    def _sift_up(self, i: int) -> None:
        """
        Restore the heap property by sifting up the element at index i

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Args:
            i: the index of the element to sift up
        """
        while i > 0 and self._heap[self._parent(i)] > self._heap[i]:
            self._swap(i, self._parent(i))
            i = self._parent(i)

    def _sift_down(self, i: int) -> None:
        """
        Restore the heap property by sifting down the element at index i

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Args:
            i: the index of the element to sift down
        """
        min_index = i
        l = self._left(i)  # left child
        if (
            l < self._size and self._heap[l] < self._heap[min_index]
        ):  # if left child is smaller than parent
            min_index = l  # set left child as new minimum
        r = self._right(i)
        if (
            r < self._size and self._heap[r] < self._heap[min_index]
        ):  # if right child is smaller than parent
            min_index = r  # set right child as new minimum
        if i != min_index:  # if the parent is not the minimum
            self._swap(i, min_index)  # swap the parent with the minimum
            self._sift_down(min_index)  # sift down the parent

    def insert(self, value: int) -> None:
        """
        Insert a new element into the heap, and restore the heap property

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Args:
            value: the value of the new element
        """

        self._heap.append(value)
        self._size += 1
        self._sift_up(self._size - 1)

    def extract_min(self) -> int:
        """
        Extract the minimum element from the heap, and restore the heap property

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Returns:
            the minimum element
        """
        result = self._heap[0]
        self._heap[0] = self._heap[self._size - 1]
        self._size -= 1
        self._sift_down(0)
        return result

    def remove(self, i: int) -> None:
        """
        Remove an element from the heap, and restore the heap property

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Args:
            i: the index of the element to remove
        """
        self._heap[i] = float("-inf")
        self._sift_up(i)
        self.extract_min()

    def change_priority(self, i: int, value: int) -> None:
        """
        Change the priority of an element in the heap, and restore the heap property

        Complexity:
            - Time: O(log n)
            - Space: O(1)

        Args:
            i: the index of the element to change
            value: the new value of the element
        """
        old_value = self._heap[i]
        self._heap[i] = value
        if value < old_value:
            self._sift_up(i)
        else:
            self._sift_down(i)

    def heapify(self, array: List[int]) -> None:
        """
        Build a heap from an array in O(n) time

        Complexity:
            - Time: O(n)
            - Space: O(n)

        Args:
            array: the array to be converted into a heap
        """
        self._heap = array
        self._size = len(array)
        for i in range(self._size // 2, -1, -1):
            self._sift_down(i)

    def to_graphviz(self) -> graphviz.Digraph:
        dot = graphviz.Digraph(comment="Heap")
        for i in range(self._size):
            dot.node(str(i), str(self._heap[i]))
            if i > 0:
                dot.edge(str(self._parent(i)), str(i))
        return dot

    def compute_height(self) -> int:
        height = 0
        i = 0
        while i < self._size:
            i = self._left(i)
            height += 1
        return height

    def is_heap(self) -> bool:
        for i in range(self._size):
            if self._left(i) < self._size and self._heap[i] > self._heap[self._left(i)]:
                return False
            if (
                self._right(i) < self._size
                and self._heap[i] > self._heap[self._right(i)]
            ):
                return False
        return True

    def __iter__(self) -> Generator[int, None, None]:
        for i in range(self._size):
            yield self._heap[i]

    def __getitem__(self, i: int) -> int:
        return self._heap[i]

    def __len__(self):
        return self._size

    def __str__(self):
        return str(self._heap[: self._size])

    def __repr__(self):
        return str(self._heap[: self._size])


def heapsort(array: List[int]) -> Tuple[List[int], Heap]:
    """
    Sort an array using a heap, this sort algorithm is not stable, but it is in-place

    Complexity:
        - Time: O(n log n)
        - Space: O(n)

    Args:
        array: the array to be sorted

    Returns:
        the sorted array
    """
    heap = Heap(array)
    for i in range(len(array) - 1, -1, -1):
        array[i] = heap.extract_min()
    return array, heap