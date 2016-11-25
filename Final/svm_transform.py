import numpy as np
from HW8.number_svm import NumberSVM

def x_matrix():
    """
    Returns the x matrix from problem 11
    :return: x matrix
    """
    x = np.array([[1, 0],
                  [0, 1],
                  [0, -1],
                  [-1, 0],
                  [0, 2],
                  [0, -2],
                  [-2, 0]], dtype='d')
    return x

def y_matrix():
    """
    Returns y matrix corresponding to x_matrix
    :return: y matrix
    """
    x = np.array([[-1],
                  [-1],
                  [-1],
                  [1],
                  [1],
                  [1],
                  [1]], dtype='d')
    return x


def transform(x):
    """
    Takes a matrix with points = {x1, x2} to the z space {x2^2 - 2x1 - 1, x1^2 -2x2 + 1}
    :param x: matrix x formatted as stated above.
    :return: z matrix
    """
    z = x
    for i, point in enumerate(x):
        x1 = point[0]
        x2 = point[1]
        z[i][0] = x2 ** 2 - 2 * x1 - 1
        z[i][1] = x1 ** 2 - 2 * x2 + 1
    return z


z = transform(x_matrix())
print(z * y_matrix())