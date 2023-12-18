from typing import TypeVar, Generic, List

T = TypeVar("T")


class BaseQueue(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Capacity must be greater than 0")
        self._size = -1
        self._capacity = capacity
        self._container: List[T] = [None] * capacity

    def push(self, item: T, idx: int) -> None:
        if self.is_full():
            raise ValueError("Full")
        self._size += 1
        self._container[idx] = item

    def pop(self, idx: int) -> T:
        if self.is_empty():
            raise ValueError("Empty")
        temp = self._container[idx]
        self._container[idx] = None
        self._size -= 1
        return temp

    def __repr__(self) -> str:
        return repr(self._container)

    def __len__(self) -> int:
        return len(self._container)

    def is_full(self) -> bool:
        return self._size == self._capacity - 1

    def is_empty(self) -> bool:
        return self._size == -1

    def make_empty(self) -> None:
        self._size = -1
        self._container = [None] * self._capacity


class ListStack(BaseQueue[T]):
    """
    LIFO (Last In First Out)

    | 3 |
    | 2 | -> pop() -> 3
    | 1 |
    |---|

    Complexity:
        - Space: O(n)
        - Push: O(1)
        - Pop: O(1)

    Usage example:
    ```python
    s = Stack[int]()
    s.push(1)
    s.push(2)
    s.push(3)
    print(s.pop())
    print(s.pop())
    print(s.pop())
    print(s.pop())
    ```
    """

    def pop(self) -> T:
        return BaseQueue.pop(self, self._size)

    def push(self, item: T) -> None:
        return BaseQueue.push(self, item, self._size + 1)


class ListQueue(BaseQueue[T]):
    """
    FIFO (First In First Out)

       |-----------|
    ->   3 | 2 | 1    -> pop() -> 1
       |-----------|

    Complexity:
        - Space: O(n)
        - Push: O(n)
        - Pop: O(1)

    Usage example:
    ```python
    q = Queue[int]()
    q.push(1)
    q.push(2)
    q.push(3)
    print(q.pop())
    print(q.pop())
    print(q.pop())
    print(q.pop())
    ```
    """

    def pop(self) -> T:
        temp = self._container[0]
        for i in range(self._size):
            self._container[i] = self._container[i + 1]
        self._container[self._size] = None
        self._size -= 1
        return temp

    def push(self, item: T) -> None:
        if self.is_full():
            raise ValueError("Full")
        self._size += 1
        self._container[self._size] = item
