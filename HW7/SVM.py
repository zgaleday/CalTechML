"""
Methods used to compare the performance of PLA and hard SVM on linearly separable data.
Dependencies on numpy and cvxopt (QP solver)
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from HW1.data_gen import DataSet
from HW1.percept_learning import pla


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

    return matrix(n_by_n)

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


def generate_min_vector(n):

    """
    Generates a zero vector of size n + 2
    :param n: number of points
    :return: zero vector on len n + 2 as cvxopt matrix
    """
    return matrix(np.array(np.zeros(n+2)))

def quad_solve(quad_matrix, linear_coef, constraints, min_vector):

    """
    Does the quadratic minimization of alpha in hard SVM using quadratic matrix and linear coefficients as input.
    Subject to linear constraints: y.T alpha = 0 and 0 <= alpha <= infinity
    :param quad_matrix: NxN matrix (output from generate matrix (1/2 Q from the equation)
    :param linear_coef: the linear coefficients (-1^T vector or linear coefficient)
    :param constraints: constraint matrix for hard SVM (y.T alpha = 0 and a >= 0 (negative identity matrix)
    :param min_vector: the minimization vector
    :return: minimized alpha vector subject to the constrains defined by hard SVM (in real N-dimensional space)
    """
    solvers.options['show_progress'] = False
    alpha = solvers.qp(quad_matrix, linear_coef, constraints, min_vector)
    alpha_array = []
    for a in alpha['x']:
        alpha_array.append(a)
    return clean_alpha(np.array(alpha_array))


def clean_alpha(alpha):

    """
    Makes non SVs go to zero
    :param alpha: alpha output from QP
    :return: cleaned alpha vector
    """
    for i, a in enumerate(alpha):
        if a < 0.001:
            alpha[i] = 0
    return alpha



def solver_ws(min_alpha, points, classifications):

    """
    Solves for the w using the minimized alpha vector from quad_solve
    :param min_alpha: result of quad solve (N-dimensional vector)
    :param points: points SVM on
    :param classifications: classifications of points {-1, 1}
    :return: w vector (N-dimensional) resultant from SVM (normalized)
    """
    w = np.zeros(len(points[0]))
    for i, alpha in enumerate(min_alpha):
        w += alpha * classifications[i] * points[i]
    return w

def svi(min_alpha):

    """
    Returns the index of a support vector
    :param min_alpha: the minimized alpha from quad_solver
    :return: index in points of a support vector
    """
    return np.argmax(min_alpha)


def solver_b(alpha, points, classifications, w):

    """
    Solves for b in equation y_n(w.Tx_n + b) = 1
    :param sv_index: index in points of a support vector
    :param points: points solving on
    :param classifications: classification vector (N-dimensional all entries {-1, 1}
    :param w: w vector resultant from hard SVM
    :return: value of bias in hard SVM
    """
    b = 0.0
    for index, a in enumerate(alpha):
        if a != 0:
            b += ((1 - classifications[index] * np.dot(w.T, points[index])) / classifications[index])
    return b / np.count_nonzero(alpha)


def classification_error(data_set, points, g):

    """
    Determines the classification error under a given hypothesis g on a set of points formatted as above (1, x1, ....)
    with give classification vector
    :param points: points to determine classification error on
    :param classifications: vector of classifications {-1, +1}
    :param g: hypothesis vector
    :return: fraction of points misclassified
    """
    misclassified = 0.0
    for index, point in enumerate(points):
        if not data_set.compare(point, g):
            misclassified += 1
    return misclassified / len(points)


def check_kkt(alpha, classifications, w, points, bias):
    for i in range(len(alpha)):
        kkt = alpha[i] * (classifications[i] * (np.dot(w.T, points[i]) + bias) - 1)
        print(kkt)


def get_SVM_hypoth(data_set, test=False):

    """
    Returns a g vector compatable with DataSet methods
    :param data_set: data set of points to train on
    :return: g vector from SVM [w_1, w_2, b], number of support vectors
    """
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
    constraints = generate_svm_constraints(classifications)
    min_vector = generate_min_vector(len(strip_points))
    alpha = quad_solve(quad, lin, constraints, min_vector)
    if np.count_nonzero(alpha) == 0:
        return None, None
    for i, a in enumerate(alpha):
        if a != 0:
            data_set.support_vector[i] = True
    w = solver_ws(alpha, strip_points, classifications)
    b = solver_b(alpha, strip_points, classifications, w)
    if test:
        check_kkt(alpha, classifications, w, strip_points, b)
    g = np.empty(3)
    g[0] = w[0]
    g[1] = w[1]
    g[2] = b
    return g, np.count_nonzero(alpha)


def compare_svm_pla(size, iters):

    """
    Method to compare the out of sample preformance of SVM and PLA
    :param size: number of data points in training set
    :param iters: number of trials
    :return: fraction of runs where svm better than pla (prints avg number of support vectors to console)
    """
    data_set = DataSet(size)
    out_set = DataSet(1000)
    svm_wins = 0.0
    avg_svs = 0.0
    i = 0
    while i < iters:
        w = np.array([0.0, 0.0, 0.0])
        if i % 10 == 0:
            print(i)
        g_svm, support_vecs = get_SVM_hypoth(data_set)
        if g_svm is None:
            data_set.new_set()
            continue
        g_pla = pla(w, data_set, toggle="vector")
        avg_svs += support_vecs
        error_pla = classification_error(data_set, out_set.points, g_pla)
        error_svm = classification_error(data_set, out_set.points, g_svm)
        if error_svm < error_pla:
            svm_wins += 1
        data_set.new_set()
        i += 1
    print(avg_svs / iters)
    return svm_wins / iters


def toysvm():
    def to_matrix(a):
        return matrix(a, tc='d')
    X = np.array([
        [0,2],
        [2,2],
        [2,0],
        [3,0]
        ])
    y = np.array([-1,-1,1,1])
    Qd = np.array([
        [0,0,0,0],
        [0,8,-4,-6],
        [0,-4,4,6],
        [0,-6,6,9],
        ])
    Ad = np.array([
        [-1,-1,1,1],
        [1,1,-1,-1],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ])
    N = len(y)
    P = to_matrix(Qd)
    q = to_matrix(-(np.ones((N))))
    G = to_matrix(-Ad)
    h = to_matrix(np.array(np.zeros(N+2)))
    alpha = quad_solve(P, q, G, h)
    w = solver_ws(alpha, X, y)
    b = solver_b(alpha, X, y, w)
    print("alpha: ", alpha)
    print("w vector: ", w)
    print("b vector: ", b)

def test_svm():
    data_set = DataSet(10)
    g_svm, support_vecs = get_SVM_hypoth(data_set, test=False)
    g_pla = pla(np.array([0.0, 0.0, 0.0]), data_set, toggle="vector")
    data_set.visualize_hypoth(g_svm)
    data_set.visualize_hypoth(g_pla)


toysvm()