from __future__ import annotations
import random
from typing import List, Generic, TypeVar, Optional, Type, Tuple

K = TypeVar("K")
V = TypeVar("V")


# HashFunctions
class BaseHashFunction(Generic[K]):
    def set_table_size(self, table_size: int):
        self.table_size = table_size

    def get_hash(self, key: K) -> int:
        pass

    def copy(self) -> BaseHashFunction[K]:
        return self.__class__()


class ModuloHashFunction(BaseHashFunction[K]):
    def get_hash(self, key: K) -> int:
        return key % self.table_size


class SecondaryHashFunction(BaseHashFunction[K]):
    @property
    def q(self) -> int:
        return self._q

    def __init__(self, q: int):
        self._q = q

    def get_hash(self, key: K) -> int:
        return self._q - (key % self._q)

    def copy(self) -> SecondaryHashFunction[K]:
        return SecondaryHashFunction(self._q)


## CollisionHandlers
class BaseCollisionHandler(Generic[K]):
    def __init__(self):
        self.reset()

    def set_table_size(self, table_size: int):
        self.table_size = table_size

    def add_collision(self):
        self.collisions += 1

    def reset(self):
        self.collisions = 0

    def handle_collision(self, hash_key: int, key: K) -> int:
        pass

    def copy(self) -> BaseCollisionHandler[K]:
        return self.__class__()


class LinearCollisionHandler(BaseCollisionHandler[K]):
    """
    Linear collision handler.
    The hash key is incremented by 1.
    If load_factor > 0.5, there is a risk of clustering.
    """

    def handle_collision(self, hash_key: int, key: K) -> int:
        self.add_collision()
        return (hash_key + 1) % self.table_size


class QuadraticCollisionHandler(BaseCollisionHandler[K]):
    def handle_collision(self, hash_key: int, key: K) -> int:
        self.add_collision()
        return (hash_key + self.collisions**2) % self.table_size


class RandomCollisionHandler(BaseCollisionHandler[K]):
    """
    Random collision handler.
    A random number is added to the hash key.

    Complexity for insertion in average:
        O(1 / (1 - load_factor)) in average
    """

    def handle_collision(self, hash_key: int, key: K) -> int:
        self.add_collision()
        return (hash_key + random.randint(1, self.table_size - 1)) % self.table_size


class DoubleHashingCollisionHandler(BaseCollisionHandler[K]):
    """
    Double hashing collision handler.
    The hash function is used to generate a new hash key.

    Complexity:
        O(1) average
        O(n) worst case (where n is the table size)
    Args:
        hash_function (BaseHashFunction): The hash function to use.
    """

    def __init__(self, hash_function: BaseHashFunction[K]):
        self.hash_function = hash_function
        self.reset()

    def set_table_size(self, table_size: int):
        self.table_size = table_size
        self.hash_function.set_table_size(table_size)

    def handle_collision(self, hash_key: int, key: K) -> int:
        value = (hash_key + self.hash_function.get_hash(key)) % self.table_size
        self.add_collision()
        return value

    def copy(self) -> DoubleHashingCollisionHandler[K]:
        return DoubleHashingCollisionHandler(self.hash_function.copy())


## HashTable
class HashTable(Generic[K, V]):
    """
    HashTable implementation with a fixed size.
    The hash function and collision handler can be set.

    The load factor is the ratio of the number of elements to the table size. It will affect the performance of the hash table.

    Complexity:
        O(1) average
        O(n) worst case (where n is the table size, when there's a collision for every key inserted)
    """

    @property
    def collisions(self) -> int:
        return self._collisions

    @property
    def size(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def load_factor(self) -> float:
        """
        The load factor is the ratio of the number of elements to the table size.
        It will affect the performance of the hash table.
        """
        return self._size / self._capacity

    def __init__(
        self,
        capacity: int,
        hash_function: BaseHashFunction,
        collision_handler: BaseCollisionHandler,
    ):
        self._capacity = capacity
        self._table: List[Optional[Tuple[int, K]]] = [(None, None)] * capacity
        self._hash_function = hash_function
        self._collision_handler = collision_handler
        self._hash_function.set_table_size(capacity)
        self._collision_handler.set_table_size(capacity)
        self._collisions = 0
        self._size = 0

    def resize(self, new_capacity: int, verbose: bool = False) -> HashTable[K, V]:
        """
        Resize the hash table.

        Args:
            new_capacity (int): The new capacity of the hash table.

        Returns:
            HashTable: The hash table.
        """
        new_capacity = max(new_capacity, self._capacity)
        new_hash_function = self._hash_function.copy()
        new_collision_handler = self._collision_handler.copy()
        new_hash_function.set_table_size(new_capacity)
        new_collision_handler.set_table_size(new_capacity)

        new_hash_table = HashTable(
            new_capacity, new_hash_function, new_collision_handler
        )

        for key, value in self._table:
            if key is not None:
                new_hash_table.insert(key, value, verbose=verbose)

        return new_hash_table

    def _is_empty(self, hash_key: int) -> bool:
        return self._table[hash_key][0] is None

    def get_hash(self, key: K) -> int:
        return self._hash_function.get_hash(key)

    def insert(self, key: K, value: V, verbose: bool = False) -> HashTable[K]:
        """
        Insert a value at the given key.
        If the key is already in use, the collision handler is used to find a new key.

        Args:
            key (int): The key to insert the value at.
            value (int): The value to insert.
        """
        if self.load_factor > 0.5:
            new_capacity = self._capacity * 2
            if verbose:
                print(
                    f"Load factor is {self.load_factor}, resizing to {new_capacity}..."
                )
            self = self.resize(new_capacity, verbose=verbose)

        self._collision_handler.reset()
        hash_key = self.get_hash(key)
        if verbose:
            print(f"Trying to insert {key} at {hash_key}...")
        while not self._is_empty(hash_key):
            if self._table[hash_key][0] == key:
                # Update the value if the key is already in use
                if verbose:
                    print(f"Updating {key} at {hash_key}...")
                break
            old_hash_key = hash_key
            hash_key = self._collision_handler.handle_collision(hash_key, key)
            if verbose:
                print(f"Collision at {old_hash_key}, trying at {hash_key}...")
        if verbose:
            print(f"Inserting {key} at {hash_key}...")
        self._collisions += self._collision_handler.collisions
        self._table[hash_key] = (key, value)
        self._size += 1
        return self

    def get(self, key: K, verbose: bool = False) -> Optional[Tuple[int, int, K]]:
        """
        Get the value at the given key.

        Args:
            key (int): The key to get the value from.

        Returns:
            Optional[int]: The value at the key or None if the key is not in use.
        """
        hash_key = self.get_hash(key)
        if verbose:
            print(f"Trying to get value at {hash_key}...")
        while not self._is_empty(hash_key):  # While the cell is not empty
            if self._table[hash_key][0] == key:  # If the key matches
                key, value = self._table[hash_key]
                return value, key, hash_key  # Return the value
            hash_key = self._collision_handler.handle_collision(
                hash_key, key
            )  # Else, try the next cell corresponding to the collision handler
        return None  # If the key is not found, return None

    def remove(self, key: K, verbose: bool = False) -> Optional[Tuple[int, int, K]]:
        """
        Remove the value at the given key.

        Args:
            key (int): The key to remove the value from.

        Returns:
            Optional[int]: The value at the key or None if the key is not in use.
        """
        hash_key = self.get_hash(key)
        if verbose:
            print(f"Trying to remove value at {hash_key}...")
        while not self._is_empty(hash_key):
            if self._table[hash_key][0] == key:
                key, value = self._table[hash_key]
                self._table[hash_key] = (None, None)
                self._size -= 1
                return value, key, hash_key
            hash_key = self._collision_handler.handle_collision(hash_key, key)
        return None

    def __str__(self) -> str:
        max_number_length = len(str(self._capacity))
        padding_length = max_number_length + 1
        keys = [
            f"{key:>{padding_length}}" if key is not None else f"{'':>{padding_length}}"
            for key, _ in self._table
        ]
        values = [
            f"{value:>{padding_length}}"
            if value is not None
            else f"{'':>{padding_length}}"
            for _, value in self._table
        ]
        indices = [f"{i:>{padding_length}}" for i in range(self._capacity)]
        return f"{' '.join(values)}\n{' '.join(keys)}\n{' '.join(indices)}"

    def __repr__(self) -> str:
        return self.__str__()
