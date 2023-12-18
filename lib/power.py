class Power:
    def elevate_1(self, x: int, n: int) -> int:
        """
        Elevates a number to a power.

        Complexity:
            Time: O(log(n))
            Space: O(1)
        Args:
            x (int): The number to elevate.
            n (int): The power to elevate to.
        Returns:
            int: The number elevated to the power.
        """
        r = 1.0
        while n > 0:
            if n % 2 == 1:
                r *= x
            x *= x
            n //= 2
        return r
