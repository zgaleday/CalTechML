"""
Methods used to compare the performance of PLA and hard SVM on linearly separable data.
Dependencies on numpy and cvxopt (QP solver)
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers


def generate_matrix(points, classifications):

    """
    Generates the matrix to be used in the quadratic term of hard SVM
    :param points: data points of size N
    :param classifications: classifications of points on target {-1, 1}
    :return: N x N matrix for use in QP minimization
    """
    n = len(points)
    n_by_n = np.array((n, n))
    for i in range(n):
        for j in range(n):
            n_by_n[i][j] = classifications[i] * classifications[j] * np.dot(points[i].T, points[j])
    return matrix(n_by_n)

def linear_coefficient(N):

    """
    Generates the vector of negative 1s to be used as the linear coefficient in QP solver
    :param N: dimension of vector
    :return: vector of negative 1s in N dimensional
    """
    return matrix(-1.0, (1, N))


def generate_svm_constraints(classifications):

    """
    Function to generate the constraint matrix for QP solution with conditions y.T alpha = 0 and alpha >= 0
    :param classifications: classification vector
    :return: n + 2 x n matrix
    """
    n = len(classifications)
    constraints = np.array((n + 2, n))
    constraints[0] = classifications.T
    constraints[1] = -1 * classifications.T
    bottom = np.identity(n, float)
    for i in range(n):
        constraints[i + 2] = -1 * bottom[i]

    return matrix(constraints)


def quad_solve(quad_matrix, linear_coef, constraints):

    """
    Does the quadratic minimization of alpha in hard SVM using quadratic matrix and linear coefficients as input.
    Subject to linear constraints: y.T alpha = 0 and 0 <= alpha <= infinity
    :param quad_matrix: NxN matrix (output from generate matrix
    :param linear_coef: the linear coefficients
    :param constraints: contraint matrix for hard SVM (y.T alpha = 0 and a >= 0)
    :return: minimized alpha vector subject to the constrains defined by hard SVM (in real N-dimensional space)
    """
    # TODO: Implement quadratic solving method


def solver_ws(min_alpha, points, classifications):

    """
    Solves for the w using the minimized alpha vector from quad_solve
    :param min_alpha: result of quad solve (N-dimensional vector)
    :param points: points SVM on
    :param classifications: classifications of points {-1, 1}
    :return: w vector (N-dimensional) resultant from SVM
    """
    # TODO: Implement solver_ws function


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


