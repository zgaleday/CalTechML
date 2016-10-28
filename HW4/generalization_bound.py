import numpy as np
import matplotlib.pyplot as plt

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
    Calculates the omega of the generalization bound using delta and simple growth under VC bound.
    :param n: see delta and simple_growth
    :param dvc: set delta
    :param d: value of the delta function
    :return: the omega value of the generalization bound
    """
    log = np.log(4 * simple_growth(2 * n, dvc) / d)
    return np.sqrt(8.0 / n * log)

def rp_bound(n, dvc, d):

    """
    Calculates the omega of the generalization bound using delta and simple growth under rp bound.
    :param n: see delta and simple_growth
    :param dvc: set delta
    :param d: value of the delta function
    :return: the omega value of the generalization bound
    """
    log1 = np.log(2.0 * simple_growth(n, dvc))
    log2 = np.log(1.0 / d)
    return np.sqrt(2.0 * log1 / n) + np.sqrt(2.0 / n * log2) + 1.0 / n

def pvb_bound(n, dvc, d):
    """
        Calculates the epsilon of the generalization bound using delta and simple growth pvb bound.
        :param n: see delta and simple_growth
        :param dvc: set delta
        :param d: value of the delta function
        :return: the roots of epsilon
        """
    log = np.log(6.0 * simple_growth(2 * n, dvc) / d)
    b = -1.0 / n * 2
    c = -1.0/ n * log
    rts = np.roots([1, b, c])
    for root in rts:
        if root > 0:
            return root


def devroye_bound(n, dvc, d):
    """
        Calculates the omega of the generalization bound using delta and simple growth devroye bound.
        :param n: see delta and simple_growth
        :param dvc: set delta
        :param d: value of the delta function
        :return: the roots of epsilon of the generalization bound
        """
    log = np.log(4.0) + dvc * np.log(n ** 2) - np.log(d)
    a = 1.0 - 2.0 / n
    b = -2.0 / n
    c = -log / (2.0 * n)
    rts = np.roots([a, b, c])
    for root in rts:
        if root > 0:
            return root

def plot_bound():
    vc = np.empty(10000)
    rp = np.empty(10000)
    pvb = np.empty(10000)
    devroye = np.empty(10000)
    for n in range(1, 10000):
        vc[n] = vc_bound(n, 50, 0.05)
        rp[n] = rp_bound(n, 50, 0.05)
        pvb[n] = pvb_bound(n, 50, 0.05)
        devroye[n] = devroye_bound(n, 50, 0.05)
    plt.axis([0, 10, 0, 20])
    plt.plot(vc, color='b', label="vc")
    plt.plot(rp, color='r', label="rp")
    plt.plot(pvb, color='black', label="pvb")
    plt.plot(devroye, color='purple', label="devroye")
    plt.legend()
    plt.show()


plot_bound()
print(vc_bound(5, 50, 0.05))
print(rp_bound(5, 50, 0.05))
print(pvb_bound(5, 50, 0.05))
print(devroye_bound(5, 50, 0.05))