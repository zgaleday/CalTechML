"""
Methods used to compare the performance of PLA and hard SVM on linearly separable data.
Dependencies on numpy and cvxopt (QP solver)
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from HW1.data_gen import DataSet


def generate_matrix(points, classifications):

    """
    Generates the matrix to be used in the quadratic term of hard SVM
    :param points: data points of size N
    :param classifications: classifications of points on target {-1, 1}
    :return: N x N matrix for use in QP minimization
    """
    n = len(points)
    n_by_n = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            n_by_n[i][j] = classifications[j] * classifications[i] * np.dot(points[j].T, points[i])
    return matrix(n_by_n)

def linear_coefficient(N):

    """
    Generates the vector of negative 1s to be used as the linear coefficient in QP solver
    :param N: dimension of vector
    :return: vector of negative 1s in N dimensional
    """
    ones = np.ones(N)
    ones *= -1
    return matrix(ones)


def generate_svm_constraints(n):

    """
    Function to generate the constraint matrix for QP solution with conditions y.T alpha = 0 and alpha >= 0
    :param n: number of data points
    :return: n + 2 x n matrix
    """
    constraints = np.negative(np.identity(n, float))

    return matrix(constraints)


def quad_solve(quad_matrix, linear_coef, constraints, classifications):

    """
    Does the quadratic minimization of alpha in hard SVM using quadratic matrix and linear coefficients as input.
    Subject to linear constraints: y.T alpha = 0 and 0 <= alpha <= infinity
    :param quad_matrix: NxN matrix (output from generate matrix
    :param linear_coef: the linear coefficients
    :param constraints: constraint matrix for hard SVM (y.T alpha = 0 and a >= 0)
    :param classifications: classification vector (len n
    :return: minimized alpha vector subject to the constrains defined by hard SVM (in real N-dimensional space)
    """
    min_vector = matrix(np.zeros((len(classifications))), tc='d')
    print(min_vector)
    alpha = solvers.qp(quad_matrix, linear_coef, constraints, min_vector, matrix(classifications.T), matrix([0.0]))
    return alpha['x']


def solver_ws(min_alpha, points, classifications):

    """
    Solves for the w using the minimized alpha vector from quad_solve
    :param min_alpha: result of quad solve (N-dimensional vector)
    :param points: points SVM on
    :param classifications: classifications of points {-1, 1}
    :return: w vector (N-dimensional) resultant from SVM
    """
    w = np.empty(2)
    for i, alpha in enumerate(min_alpha):
        w = np.add(alpha, np.multiply(classifications[i], points[i]))
    return w

def support_vector_index(min_alpha):

    """
    Returns the index of a support vector
    :param min_alpha: the minimized alpha from quad_solver
    :return: index in points of a support vector
    """
    # TODO: Implement support vector index function


def solver_b(sv_index, points, classifications, w):

    """
    Solves for b in equation y_n(w.Tx_n + b) = 1
    :param sv_index: index in points of a support vector
    :param points: points solving on
    :param classifications: classification vector (N-dimensional all entries {-1, 1}
    :param w: w vector resultant from hard SVM
    :return: value of bias in hard SVM
    """
    # TODO: Implement solver for bias in hard SVM


def error(g, data_set):

    """
    Determines the fraction of misclassified points in a given data set under a given hypothesis
    :param g: hypothesis vector (with constant included)
    :param data_set: DataSet object
    :return: fraction of points misclassified under current set
    """
    # TODO: Implement error function


data_set = DataSet(10)
strip_points = np.empty((len(data_set.points), 2))
for index, point in enumerate(data_set.points):
    strip_points[index][0] = point[0]
    strip_points[index][1] = point[1]
classifications = np.empty((len(data_set.bools), 1))
for i, bool in enumerate(data_set.bools):
    if bool:
        classifications[i][0] = 1.0
    else:
        classifications[i][0] = -1.0
quad = generate_matrix(strip_points, classifications)
lin = linear_coefficient(len(strip_points))
constraints = generate_svm_constraints(len(strip_points))
alpha = quad_solve(quad, lin, constraints, classifications)
w = solver_ws(alpha, strip_points, classifications)
print(w)