import numpy as np
import matplotlib as plt

def delta(n, epsilon, dvc):

    """
    Calculated the delta constant in the generalization bound using simple_growth
    :param n: Number of points (> 0)
    :param epsilon: epsilon of generalization function
    :param dvc: see simple growth
    :return: the delta value in generalization bound
    """
    exponent = np.exp((-1 * (1.0 / 8.0)) * (epsilon ** 2) * n)
    return 4 * simple_growth(2 * n, dvc) * exponent


def simple_growth(n, dvc):

    """
    A simplified growth function for the generalization bound m_h(N) = N^dvc.
    :param n: Number of points in the data set (> 0)
    :param dvc: The VC dimension of the hypothesis set (> 0)
    :return: value of simplified growth function
    """
    return n ** dvc


def vc_bound(n, dvc, d):

    """
    Calculates the omega of the generalization bound using delta and simple growth.
    :param n: see delta and simple_growth
    :param dvc: set delta
    :param d: value of the delta function
    :return: the omega value of the generalization bound
    """
    log = np.log(4 * simple_growth(2 * n, dvc) / d)
    return np.sqrt(8.0 / n * log)

def rp_bound(n, dvc, d):

    """
    Calculates the omega of the generalization bound using delta and simple growth.
    :param n: see delta and simple_growth
    :param dvc: set delta
    :param d: value of the delta function
    :return: the omega value of the generalization bound
    """
    log1 = np.log(2 * simple_growth(n, dvc))
    log2 = np.log(1/d)
    return np.sqrt(2 * log1 / n) + np.sqrt(2 / n *log2) + 1 / n




