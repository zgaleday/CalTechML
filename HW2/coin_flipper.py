import numpy as np


"""Metod generated 10 samples for a given number of coins.  Outputs and array of the number of heads achieved
by each coin
Params: number of coins to flip.
Return: Array with number of heads.
"""


def ten_flips(number):

    return np.random.binomial(10, 0.5, 1000)


"""
Returns the minimum number of heads for a give trail of 10 flips
Params: An array of ints
Return: The minimum value in that array ($C_min$)
"""
def min_heads(result):
    return np.amin(result)


"""
Returns the number of heads of a random coin in the sample
Params: An array of ints
Return: The number of heads of a random coin in the sample.
"""
def rand_heads(result):
    return result[np.random.randint(0, len(result))]


"""
A method to run 10 flip experiment 100,000 times.  Returns the avg value for c_1, c_rand, and c_min
Params: number of coins
Return: c_1, c_rand, and c_min avg val
"""
def repeats(number):
    c_1 = 0.0
    c_rand = 0.0
    c_min = 0.0
    for trial in range(1000):
        result = ten_flips(number)
        c_1 = (c_1 * trial + result[1]) / (trial + 1)
        c_rand = (c_rand * trial + rand_heads(result)) / (trial + 1)
        c_min = (c_min * trial + min_heads(result)) / (trial + 1)

    return c_1, c_rand, c_min


print(repeats(1000))

