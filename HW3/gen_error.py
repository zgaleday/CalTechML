import numpy as np

def Hoeffding(m, n):

    """
    Calculates Hoeffding's inequality at given n and m

    :param m: Number of hypotheses (>= 0)
    :param n: Number of points in sample (>= 0)
    :return: Value of Hoeffding's inequality at given n and m
    """

    eps = 0.05
    return 2 * m * np.exp(-2 * (eps ** 2) * n)

list = [500, 1000, 1500, 2000]
for number in list:
    print(number)
    print(Hoeffding(100, number))