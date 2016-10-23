import numpy as np
import scipy.integrate as integrate

def sine_compare(x, y):

    """
    Classifies points compared to sine function
    :param x: x coord
    :param y: y coord
    :return: -1 if below or equal 1 otherwise
    """
    if np.sin(np.pi * x) <= y:
        return -1
    else:
        return 1


def avg_compare(avg, x, y):
    """
    Classifies points compared to avg * x function
    :param x: x coord
    :param y: y coord
    :return: -1 if below or equal 1 otherwise
    """
    if avg * x <= y:
        return -1
    else:
        return 1

def linear_regression(points):

    """
    A method to computed the least squares linear regression for classification for ax target.

    Params: two sets of x,y coords in a 1x4 array
    Return: a calculated by LR.
    """
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x = np.array([[x1], [x2]])
    target = ([sine_compare(x1, y1), sine_compare(x2, y2)])
    pseudo_inverse = np.linalg.pinv(x)
    a = np.dot(pseudo_inverse, target)

    return a


def repeats(trials):

    """
    Runs linear_regression trials times recording the a values and returing an array of the values of lenght trials
    :param trials: number of trials
    :return: array of trials length with a values
    """
    result = np.empty(trials)
    for trial in range(trials):
        temp = np.random.uniform(-1, 1, 4)
        temp[1] = np.sin(np.pi * temp[0])
        temp[3] = np.sin(np.pi * temp[2])
        result[trial] = linear_regression(np.random.uniform(-1, 1, 4))

    return result

def var(result, avg):

    """
    Calculates the variance of the ax solution
    :param result: array of a's from repeats
    :param avg: average hypothesis
    :return: the expected value of the variance
    """
    expected = 0.0
    for count, a in enumerate(result):
        expected = (expected * count + (1.0 / 2.0) * integrate.quad(lambda x: (a * x - avg * x) ** 2, -1, 1)[0]) / (count + 1)
    return expected

def bias(avg):
    trials = 10000
    error = 0.0
    for trial in range(trials):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if sine_compare(x, y) != avg_compare(avg, x, y):
            error += 1
    return error / trials

a = repeats(100000)
average = np.average(a)
variance = var(a, average)
print("Average: ", average)
print("var: ", var(a, average))