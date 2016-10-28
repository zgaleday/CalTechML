import numpy as np

""""
Method for minimizing the error with error fnctn E(u,v) = (ue^v - 2ve^-u)^2 with
gradient descent using the first order linearity approximation with a learning rate
of 0.1
"""

def partial_wrt_u(u, v):

    """
    Evaluates the partial derivative wrt u for the given error function.  Returns the numerical value
    of the partial
    :param u: current weight of u as float
    :param v: current weight of v as float
    :return: Numerical eval of partial 2(e^v + 2ve^-u)(ue^v - 2ve^-u)
    """
    first_parenthesis = np.exp(v) + 2 * v * np.exp(-u)
    second_parenthesis = u * np.exp(v) - 2 * v * np.exp(-u)
    return 2 * first_parenthesis * second_parenthesis


def partial_wrt_v(u, v):

    """
    Evaluates the partial derivative wrt v for the given error function.  Returns the numerical value
    of the partial
    :param u: current weight of u as float
    :param v: current weight of v as float
    :return: Numerical eval of partial 2(ue^v - 2ve^-u)(ue^v - 2e^-u)
    """
    first_parenthesis = u * np.exp(v) - 2 * v * np.exp(-u)
    second_parenthesis = u * np.exp(v) - 2 * np.exp(-u)
    return 2 * first_parenthesis * second_parenthesis


def gradient_vector(u, v):
    """
    Method to calculate and return the gradient vector for E(u,v) at given weights
    :param u: current weight u
    :param v: current weight v
    :return: [u', v']
    """

    return [partial_wrt_u(u, v), partial_wrt_v(u, v)]