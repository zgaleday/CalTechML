import numpy as np

def delta(n, epsilon, dvc):

    """
    Calculated the delta constant in the generalization bound using simple_growth
    :param n: Number of points (> 0)
    :param epsilon: epsilon of generalization function
    :param dvc: see simple growth
    :return: the delta value in generalization bound
    """
    exponent = np.exp(-(1.0 / 8.0) * epsilon ** 2 * n)
    return 4 * simple_growth(n, dvc) * exponent


def simple_growth(n, dvc):

    """
    A simplified growth function for the generalization bound m_h(N) = N^dvc.
    :param n: Number of points in the data set (> 0)
    :param dvc: The VC dimension of the hypothesis set (> 0)
    :return: value of simplified growth function
    """
    return n ** dvc


def omega(n, epsilon, dvc):

    """
    Calculates the omega of the generalization bound using delta and simple growth.
    :param n: see delta and simple_growth
    :param epsilon: see dela
    :param dvc: see simple growth
    :return: the omega value of the generalization bound
    """
    return -1
