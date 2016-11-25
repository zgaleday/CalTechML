import numpy as np

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