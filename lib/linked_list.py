from __future__ import annotations
from typing import List, TypeVar, Generic, Optional

T = TypeVar("T")


class Node(Generic[T]):
    _value: T
    _next: Optional[Node[T]] = None
    _prev: Optional[Node[T]] = None

    @property
    def value(self):
        return self._value

    @property
    def next(self):
        return self._next

    @property
    def prev(self):
        return self._prev

    def __init__(
        self, value: T, next: Optional[Node[T]] = None, prev: Optional[Node[T]] = None
    ):
        self._value = value
        self._next = next
        self._prev = prev

    def __repr__(self):
        return str(self._value)

    def __str__(self):
        return str(self._value)


class DoublyLinkedIterator(Generic[T]):
    _current: Optional[Node[T]]

    def __init__(self, current: Optional[Node[T]] = None):
        self._current = current

    # modifie la valeur de l'élément actuellement itéré
    def set(self, e: T):
        self._current._value = e

    # retourne la valeur stockée dans l'élément actuellement itéré
    def get(self):
        return self._current.value

    # retourne une instance de l'itérateur sur le prochain élément de la liste
    # s'il y en a un
    def increment(self) -> DoublyLinkedIterator[T]:
        self._current = self._current.next
        return self

    # retourne une instance de l'itérateur sur l'élément précédent de la liste
    # s'il y en a un
    def decrement(self) -> DoublyLinkedIterator[T]:
        self._current = self._current.prev
        return self

    # retourne une valeur booléenne, selon si l'itérateur passé en paramètre énumère
    # la même liste et est positionner au même endroit.
    # Autrement dit, si les deux itérateurs sont sur le même élément
    def equals(self, o: DoublyLinkedIterator[T]) -> bool:
        return self._current == o._current


class DoublyLinkedList(Generic[T]):
    _head: Optional[Node[T]]
    _tail: Optional[Node[T]]
    _size: int

    @property
    def size(self):
        return self._size

    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    # retourne une instance de l'itérateur DoublyLinkedIterator ayant le
    # premier élément de la liste comme position initiale.
    def begin(self) -> DoublyLinkedIterator[T]:
        return DoublyLinkedIterator(self._head)

    # retourne une instance de l'itérateur DoublyLinkedIterator ayant le
    # dernier élément de la liste comme position initiale
    def end(self) -> DoublyLinkedIterator[T]:
        return DoublyLinkedIterator(self._tail)

    # ajoute un élément à la fin de la liste
    def add(self, e: T) -> Node[T]:
        n = Node(e)
        if self._head is None:
            self._head = n
            self._tail = n
        else:
            n._prev = self._tail
            self._tail._next = n
            self._tail = n
        self._size += 1
        return self._tail

    # supprime l'élément en tête de liste (le premier) et le retourne
    def remove(self) -> Node[T]:
        old_head = self._head
        self._head = old_head.next
        self._size -= 1
        return old_head.value  # retourne l'élément supprimé

    def remove_last(self) -> Node[T]:
        old_tail = self._tail
        self._tail = old_tail.prev
        self._size -= 1
        return old_tail.value

    def remove_at(self, it: DoublyLinkedIterator[T]) -> Node[T]:
        if it._current is None:
            raise ItemNotFound()
        if it._current == self._head:
            return self.remove()
        if it._current == self._tail:
            return self.remove_last()
        it._current.prev.next = it._current.next
        it._current.next.prev = it._current.prev
        self._size -= 1
        return it._current.value

    # retourne une valeur booléenne, selon que la liste est vide ou non
    def is_empty(self) -> bool:
        return self._size == 0
