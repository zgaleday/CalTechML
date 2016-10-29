import numpy as np
from HW1.data_gen import DataSet

""""
Methods to do logistic regression in the 2D plane from [-1,1]x[-1,1].
Error will be measured using the "cross entropy error"  function ln(1 +e^(-(y_n)transpose(w)x_n) at each time step
At each epoch the reg will go through each xi, yi pair in a random order.
The alg will stop when the delta w is below 0.01 (measured at the end of an epoch)
"""

def gradient(y, x, w):

    """
    Calculates the value of the gradient for a given point x_n, y_n @ a given weight
    :param y: Classification wrt to target (+ or - 1 only valid)
    :param x: Point vector [x_1, x_2, x_0]
    :param w: weight vector [w_1, w_2, w_0]
    :return: the gradient vector numerically evaluated [w_1', w_2', w_0'] as np array
    """
    return np.array([partial(y, x, w, 0), partial(y, x, w, 1), partial(y, x, w, 2)])


def partial(y, x, w, index):

    """
    Calculates the numerical value of the partial derivative wrt to the index into the weight vector.
    i.e. index 0 returns w_1', index 1 returns w_2', and index 2 returns w_0'
    :param y: Classification wrt to target (+ or - 1 only valid)
    :param x: Point vector [x_1, x_2, x_0] (Array of floats)
    :param w: weight vector [w_1, w_2, w_0] (Array of floats)
    :param index: index in weight vector to calculate the partial wrt (see above)
    :return: numerical value of the partial wrt index into w vector
    """
    exponential = np.exp(-y * np.dot(w, x))
    numerator = -y * x[index] * exponential
    denominator = 1 + exponential

    return numerator / denominator


def point_wise_gd(y, x, w, rate):

    """
    does a single timestep of gradient descent give a target value x vector and w(t). Calculates and returns w(t + 1)
    using given learning rate
    :param y: Classification wrt to target (+ or - 1 only valid)
    :param x: Point vector [x_1, x_2, x_0] (Array of floats)
    :param w: weight vector [w_1, w_2, w_0] at time t(Array of floats)
    :param rate: Learning rate (float (0.0, 1.0])
    :return: weight vector at time t + 1 (single step of gradient descent
    """
    return np.subtract(w, rate * gradient(y, x, w))