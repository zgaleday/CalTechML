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


def problem_12():
    """
    Method to solve SVM w/ poly kernel method with Q = 2
    :return: number of SVs
    """
    svm_instance = NumberSVM()
    svm_instance.X = x_matrix()
    svm_instance.Y = y_matrix().ravel()
    svm_instance.set_poly_svm_params(2, np.inf)
    svm_instance.svm_solver()
    return np.sum(svm_instance.svm.n_support_)


def coef_matrix():

    z = y_matrix() * transform(x_matrix())
    z = np.concatenate((z, y_matrix()), axis=1)
    return z


def test_system(coef_matrix, w1, w2, b):
    h = np.array([[w1], [w2], [b]])
    for coefs in coef_matrix:
        sum = np.dot(coefs, h)
        print(sum)



test_system(coef_matrix(), 1, 0, -.5)