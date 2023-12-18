class BinaryNumber:
    @staticmethod
    def to_string(n: int, bits: int = 8) -> str:
        """
        Prints the binary representation of a number.

        Complexity:
            Time: O(bits)
            Space: O(bits)
        Args:
            n (int): The number to print.
            bits (int): The number of bits.
        Returns:
            str: The binary representation of the number.
        """
        return f"{n:0{bits}b}"

class Complement2:
    def __init__(self, bits: int):
        """
        Initializes the 2's complement converter with the number of bits.

        Args:
            bits (int): The number of bits.
        """
        self._bits = bits
        self._max = 2**(bits - 1) - 1
        self._min = -2**(bits - 1)
    
    def convert(self, n: int) -> int:
        """
        Converts a number to its 2's complement representation.

        Complexity:
            Time: O(1)
            Space: O(1)
        Args:
            n (int): The number to convert.
        Returns:
            int: The 2's complement representation of the number.
        """
        if n < self._min or n > self._max:
            raise ValueError(f"n must be between {self._min} and {self._max}.")

        if n >= 0:
            return n
        else:
            return 2**self._bits + n
        
    def convert_back(self, n: int) -> int:
        """
        Converts a number from its 2's complement representation.

        Complexity:
            Time: O(1)
            Space: O(1)
        Args:
            n (int): The number to convert.
        Returns:
            int: The number from its 2's complement representation.
        """
        if n < 0 or n >= 2**self._bits:
            raise ValueError(f"n must be between 0 and {2**self._bits - 1}.")

        if n <= 2**(self._bits - 1) - 1:
            return n
        else:
            return n - 2**self._bits
