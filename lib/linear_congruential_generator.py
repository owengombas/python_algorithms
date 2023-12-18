import time
import math
from typing import List, Any


class LinearCongruentialGenerator:
    """
    Linear Congruential Generator for pseudo-random numbers.

    This class implements a pseudo-random number generator using the linear congruential method.
    The generator is defined by the recurrence relation:
    Z_(n+1) = (A * Z_n) mod M
    where Z is the sequence of pseudo-random numbers, and A, M are constants.

    Attributes:
        state (int): The current state of the generator.
        A (int): The multiplier.
        M (int): The modulus.
        Q (int): The quotient (M div A).
        R (int): The remainder (M mod A).

    Usage example:
        ```python
        generator = LinearCongruentialGenerator(seed=12345)
        random_numbers = [generator.next() for _ in range(5)]
        random_numbers
        ```
    """

    @property
    def state(self) -> int:
        """The current state of the generator."""
        return self._state

    @state.setter
    def state(self, value: int):
        self._state = value

    @property
    def A(self) -> int:
        """The multiplier."""
        return self._A

    @property
    def M(self) -> int:
        """The modulus."""
        return self._M

    @property
    def Q(self) -> int:
        """The quotient (M div A)."""
        return self._Q

    @property
    def R(self) -> int:
        """The remainder (M mod A)."""
        return self._R

    def __init__(self, seed: int, A: int = 48271, M: int = 2**31 - 1):
        """Initializes the generator with a given seed.

        Args:
            seed (int): The initial seed value.
            A (int, optional): The multiplier. Defaults to 48271.
            M (int, optional): The modulus. Defaults to 2**31 - 1.
        """
        self._M = M  # The modulus, a prime number.
        self._A = A  # The multiplier.
        self._Q = self._M // self._A  # The quotient.
        self._R = self._M % self._A  # The remainder.

        self._state = seed % self._M  # The current state of the generator.
        while self._state == 0:
            # If the state is 0, the next state will be 0 as well, so we need to avoid that.
            # We can do this by changing the seed to a different value, based on the current time.
            # So it's just like having a random seed.
            self._state = int(time.time() * 1000) % self._M
        while self._state < 0:
            # If the state is negative, we need to make it positive, we can do this by adding the modulus to it.
            self._state += self._M

    def random_integer(self) -> int:
        """
        Generates the next pseudo-random number.

        Complexity:
            Time: O(1)
            Space: O(1)
        Returns:
            int: The next pseudo-random number in the sequence.
        """
        # Decomposition into auxiliary values to prevent overflow ()
        # Z_(n+1) = (A * Z_n) mod M = (A * (Z_n % Q) - R * (Z_n // Q)) mod M
        temp = self._A * (self._state % self._Q) - self._R * (self._state // self._Q)
        if temp >= 0:
            self._state = temp
        else:
            self._state = temp + self._M
        return self._state

    def random_float(self) -> float:
        """
        Generates the next pseudo-random number.

        Complexity:
            Time: O(1)
            Space: O(1)
        Returns:
            float: The next pseudo-random number in the sequence.
        """
        return self.random_integer() / self._M

    def random_float_range(self, bound_min: float, bound_max: float) -> float:
        """
        Generates the next pseudo-random number.

        Complexity:
            Time: O(1)
            Space: O(1)
        Returns:
            float: The next pseudo-random number in the sequence.
        """
        return bound_min + self.random_float() * (bound_max - bound_min)

    def random_integer_range(self, bound_min: int, bound_max: int) -> int:
        """
        Generates the next pseudo-random number.

        Complexity:
            Time: O(1)
            Space: O(1)
        Returns:
            int: The next pseudo-random number in the sequence.
        """
        partition_size = self._M / (bound_max - bound_min + 1)
        return math.trunc(self.random_integer() / partition_size) + bound_min

    def neg_exp(self, expected_value: float) -> float:
        """
        Generates the next pseudo-random number.

        Complexity:
            Time: O(1)
            Space: O(1)
        Returns:
            float: The next pseudo-random number in the sequence.
        """
        return -expected_value * math.log(self.random_float())
