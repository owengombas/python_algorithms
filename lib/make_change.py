from typing import List, Generator, Tuple


def compute_optimial_change(
    coins: List[float], max_change: float
) -> Tuple[List[float], List[float]]:
    """
    Computes the optimal change for a given amount and a given set of coins

    Complexity:
        - Assume different_coins = sorted(set(coins))
        - Assume len(different_coins) = n
        - Assume max_change = m
        - Time: O(n*m)
        - Space: O(m)

    Args:
        coins: list of coins that can be used to make change (in a cashiers drawer, it would be the coins he has)
        max_change: amount to make change for

    Returns:
        coins_used: list of coins used to make change for amount, the indices of coins_used are the amounts
        last_coins: list of last coin used to make change for amount, the indices of last_coin are the amounts
    """
    different_coins = sorted(set(coins))  # list of different coins

    coins_used = [0] * (max_change + 1)  # list of coins used to make change for amount
    # amount=3, coins=[1,2,3] -> coins_used=[0,0,0,0]
    # This contains the part of the solution that we are building

    cents = 1  # current amount
    last_coins = [cents] * (
        max_change + 1
    )  # list of last coin used to make change for amount, initialized to 1 because we use 1 cent coins
    # amount=3, coins=[1,2,3] -> last_coin=[1,1,1,1]

    while cents <= max_change:
        min_coins = cents  # minimum number of coins to make change for amount
        new_coin = 1  # last coin used to make change for amount
        for j in range(len(different_coins)):  # for each coin
            if coins[j] > cents:  # if coin is greater than amount
                break  # stop
            if coins_used[cents - coins[j]] + 1 < min_coins:
                # if number of coins used to make change for amount - coin + 1 is less than minimum number of coins
                min_coins = (
                    coins_used[cents - coins[j]]
                    + 1  # Recall that coins_used contains the minimum number of coins used to make change for amount
                )  # update minimum number of coins by adding 1 because we used a coin
                new_coin = coins[j]  # update last coin used
        coins_used[
            cents
        ] = min_coins  # update minimum number of coins used to make change for amount for the current amount
        last_coins[
            cents
        ] = new_coin  # update last coin used to make change for amount for the current amount
        cents += 1  # update current amount by adding 1

    # Now coins_used contains the minimum number of coins used to make change for amount, the indices of coins_used are the amounts
    # Now last_coin contains the last coin used to make change for amount, the indices of last_coin are the amounts

    return coins_used, last_coins


def make_change(
    coins_used: List[float], last_coins: List[float], max_change: float
) -> Generator[Tuple[float, float, float], None, None]:
    """
    Makes change for a given amount and a given set of coins

    Yields:
        coin: coin used to make change for amount
        number_of_coins: number of coins used to make change for amount
        amount: amount for which change was made
    """

    amount = max_change
    while amount > 0:
        coin = last_coins[amount]
        yield coin, amount, coins_used[amount]
        amount -= coin


def print_change(
    coins_used: List[float], last_coins: List[float], max_change: float
) -> None:
    """
    Prints the change for a given amount and a given set of coins
    """
    for coin, amount, number_of_coins in make_change(
        coins_used, last_coins, max_change
    ):
        print(f"{number_of_coins} x {coin}")