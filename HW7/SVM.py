"""
Methods used to compare the performance of PLA and hard SVM on linearly separable data.
Dependencies on numpy and cvxopt (QP solver)
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from HW1.data_gen import DataSet
import HW1.percept_learning


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
            n_by_n[i][j] = classifications[i] * classifications[j] * np.dot(points[i].T, points[j])

    return matrix(np.multiply(0.5, n_by_n))

def linear_coefficient(N):

    """
    Generates the vector of negative 1s to be used as the linear coefficient in QP solver
    :param N: dimension of vector
    :return: vector of negative 1s in N dimensional
    """
    ones = np.ones((1, N))
    ones *= -1
    ones = matrix(ones.T)
    return ones


def generate_svm_constraints(classifications):

    """
    Function to generate the constraint matrix for QP solution with conditions y.T alpha = 0 and alpha >= 0
    :param n: number of data points
    :return: n + 2 x n matrix
    """
    n = len(classifications)
    constraints = np.empty((n + 2, n))
    constraints[0] = classifications.T
    constraints[1] = np.negative(classifications.T)
    for index, row in enumerate(np.diag(-1 * np.ones(n))):
        constraints[index + 2] = row
    return matrix(constraints, tc='d')


def quad_solve(quad_matrix, linear_coef, constraints, classifications):

    """
    Does the quadratic minimization of alpha in hard SVM using quadratic matrix and linear coefficients as input.
    Subject to linear constraints: y.T alpha = 0 and 0 <= alpha <= infinity
    :param quad_matrix: NxN matrix (output from generate matrix (1/2 Q from the equation)
    :param linear_coef: the linear coefficients (-1^T vector or linear coefficient)
    :param constraints: constraint matrix for hard SVM (y.T alpha = 0 and a >= 0 (negative identity matrix)
    :param classifications: classification vector ({-1, +1})
    :return: minimized alpha vector subject to the constrains defined by hard SVM (in real N-dimensional space)
    """
    solvers.options['show_progress'] = False
    min_vector = matrix(np.zeros(len(classifications) + 2), tc='d')         #zero vector set equal to constraints
    alpha = solvers.qp(quad_matrix, linear_coef, constraints, min_vector)['x']
    return alpha


def solver_ws(min_alpha, points, classifications, alpha_normal=False):

    """
    Solves for the w using the minimized alpha vector from quad_solve
    :param min_alpha: result of quad solve (N-dimensional vector)
    :param points: points SVM on
    :param classifications: classifications of points {-1, 1}
    :return: w vector (N-dimensional) resultant from SVM (normalized)
    """
    w = np.zeros(len(points[0]))
    for i, alpha in enumerate(min_alpha):
        mult = alpha * classifications[i]
        w = np.add(w, (mult * points[i]))
    if not alpha_normal:
        min_alpha = norm_alpha(min_alpha, points, classifications,
                               solver_b(svi(min_alpha)[0], points, classifications, w), w)
        return solver_ws(min_alpha, points, classifications, alpha_normal=True)
    else:
        return w

def svi(min_alpha):

    """
    Returns the index of two support vector
    :param min_alpha: the minimized alpha from quad_solver
    :return: index in points of two support vectors of diff classification
    """
    max_index = np.argmax(min_alpha)
    max_alpha = 0.0
    max_index2 = -1
    for index, alpha in enumerate(min_alpha):
        if alpha > max_alpha and index != max_index and classifications[max_index] != classifications[index]:
            max_index2 = index
    return max_index, max_index2


def norm_alpha(alpha, points, classifications, bias, w):

    """
    Returns the correct normalization of w s.t. support vectors are normalized to 1
    :param alpha: output vector from QP minimization
    :param points: data point
    :param classifications: classification vector {-1, +1
    :param w: w vector from sum of SVs
    :return: normalized w
    """
    svi1, svi2 = svi(alpha)
    skew = (classifications[svi1] * (np.dot(w.T, points[svi1]) + bias) - 1)
    for index, a in enumerate(alpha):
        if index == svi1:
            skew = np.multiply(a, skew)
    alpha = np.subtract(alpha, skew)
    return alpha


def solver_b(sv_index, points, classifications, w):

    """
    Solves for b in equation y_n(w.Tx_n + b) = 1
    :param sv_index: index in points of a support vector
    :param points: points solving on
    :param classifications: classification vector (N-dimensional all entries {-1, 1}
    :param w: w vector resultant from hard SVM
    :return: value of bias in hard SVM
    """
    y = classifications[sv_index]
    x = points[sv_index]
    return (1.0 - (y * np.dot(w.T, x))) / y


def error(g, data_set):

    """
    Determines the fraction of misclassified points in a given data set under a given hypothesis
    :param g: hypothesis vector (with constant included)
    :param data_set: DataSet object
    :return: fraction of points misclassified under current set
    """
    # TODO: Implement error function


def test_svm():
    data_set = DataSet(10)
    # strip_points = np.array([[1], [2], [3]])
    strip_points = np.empty((len(data_set.points), 2))
    for index, point in enumerate(data_set.points):
        strip_points[index][0] = point[0]
        strip_points[index][1] = point[1]
    classifications = np.empty((len(data_set.bools), 1))
    # classifications = np.array([[-1.0], [-1.0], [1.0]])
    for i, bool in enumerate(data_set.bools):
        if bool:
            classifications[i][0] = 1.0
        else:
            classifications[i][0] = -1.0
    quad = generate_matrix(strip_points, classifications)
    lin = linear_coefficient(len(strip_points))
    constraints = generate_svm_constraints(classifications)
    alpha = quad_solve(quad, lin, constraints, classifications)
    w = solver_ws(alpha, strip_points, classifications)
    print(alpha)
    b = solver_b(svi(alpha)[0], strip_points, classifications, w)
    # g = np.array([b[0], w[0]])
    # print(np.dot([1.0, 2], g))
    g = np.empty(3)
    g[0] = w[0]
    g[1] = w[1]
    g[2] = b
    data_set.visualize_hypoth(g)