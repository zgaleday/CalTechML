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


def run_epoch(classifications, points, w, rate):

    """
    Runs through a single epoch of SGD and returns the value of the current weight vector.
    Goes through a random permutation of the x vectors in points.
    :param classifications: an array of ints {-1, 1} for the classification of each point compared to target
    :param points: array of x vectors in [-1, 1] x [-1, 1] space and organized as discussed in gradient (depends on DataSet)
    :param w: weight vector [w_1, w_2, w_0] (Array of floats)
    :param rate: Learning rate (float (0.0, 1.0])
    :return: new weight vector after the epoch
    """
    np.random.permutation(points)
    for count, point in enumerate(points):
        w = point_wise_gd(classifications[count], point, w, rate)
    return w


def sgd(data_set, rate):

    """
    Method to run full SGD on a set of points of size points.  Stops when delta(w) < 0.01
    :param size: number of points in the data set
    :param rate: Learning rate (float (0.0, 1.0])
    :return: number of epochs to converge
    """
    classification = np.empty(data_set.size)
    for index, bool in enumerate(data_set.bools):
        if bool == False:
            classification[index] = -1
        else:
            classification[index] = 1
    epochs = 0
    w_previous = [1.0, 1.0, 1.0]
    w_current = [0.0, 0.0, 0.0]
    delta_w = np.subtract(w_previous, w_current)
    while(np.linalg.norm(delta_w) >= 0.01):
        epochs += 1
        w_previous = w_current
        w_current = run_epoch(classification, data_set.points, w_current, rate)
        delta_w = np.subtract(w_previous, w_current)
    return epochs


def average_epochs(size, rate, repeats):

    """
    Returns the average number of epochs it takes for sgd to terminate given the rule defined in the spec above
    :param size: number of points in the data set
    :param rate: Learning rate (float (0.0, 1.0])
    :param repeats: number of points in the average
    :return: the average, number of epochs to converge
    """
    data_set = DataSet(size)
    average = 0.0
    for trial in range(repeats):
        average = (average * trial + sgd(data_set, rate)) / (trial + 1)
        data_set.new_set()
    return average

print(average_epochs(100, .01, 1000))