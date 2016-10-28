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


def calculate_error(u,v):

    """
    Evaluates the error function at given weights u and v)
    :param u: current weight of u
    :param v: current weight of v
    :return: current value of the error function
    """
    parenthesis = u * np.exp(v) - 2 * v * np.exp(-u)
    return parenthesis ** 2


def gradient_descent_iterations(u, v, n, rate):

    """
    Method to evaluate the value of the error after n iterations of gradient descent starting with weight u, v with
    learning rate of rate.
    :param u: starting weight of u
    :param v: starting weight of v
    :param n: number of iterations of gradient descent
    :param rate: learning rate of gradient descent
    :return: error after last iteration of gradient descent is completed
    """
    error = calculate_error(u,v)
    for iteration in range(n):
        old_u = u
        old_v = v
        u -= (rate * partial_wrt_u(old_u, old_v))
        v -= (rate * partial_wrt_v(old_u, old_v))
        error = calculate_error(u, v)
    return error


print(gradient_descent_iterations(1.0, 1.0, 17, .1))