from typing import List


class MaxSubArraySum:
    """
    Given an array of integers, find the maximum sum of a subarray of the given array.
    For instance, given the array [-2, 1, -3, 4, -1, 2, 1, -5, 4], the maximum sum of a subarray is 6, which is the sum of [4, -1, 2, 1].

    Usage example:
        ```python
        max_sub_array_sum = MaxSubArraySum()
        max_sub_array_sum.max_sub_array_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        ```
    """

    def max_sub_array_sum(self, arr: List[int]) -> int:
        """
        Finds the maximum sum of a subarray of the given array.
        This algorithm is known as Kadane's algorithm, it uses dynamic programming.

        It works as follows:
        1) Iterate over the array.
        2) For each element, add it to the current sum.
        3) If the current sum is greater than the maximum sum, update the maximum sum.
        4) If the current sum is less than 0, reset the current sum to 0.
        
        Complexity:
            Time: O(n)
            Space: O(1)
        Args:
            arr (list): The given array.
        Returns:
            int: The maximum sum of a subarray of the given array.
        """
        max_sum = 0
        current_sum = 0
        for i in range(len(arr)):
            current_sum += arr[i]
            if current_sum > max_sum:
                max_sum = current_sum
            if current_sum < 0:
                current_sum = 0
        return max_sum
