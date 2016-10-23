import numpy as np

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


