__author__ = 'zachary'
import numpy as np
from data_gen import DataSet

"""
Takes two vectors as input.  The w vector or hypothesis vector and the misclassified point vector.
Updated the w vector by taking the vector sum of w + x to give an updated w vector.
This updated w vector is the return value.  This w results in the input vector now being correctly classified wrt hypothesis
and target function.
"""


def update(h, data_set, index):
    if data_set.bools[index]:
        d = 1
    else:
        d = -1
    temp = np.add(h, np.multiply(d, data_set.points[index]))
    return temp


"""
A method to run the PLA on the generated data set.  Hypothesis starts as the 3D Z vector.  All points misclassified
under this hypoth.  Picks a random misclassified point wrt training value and updates the hypothesis with update().
Runs until all points correctly classified under hypothesis.  Returns the hypothesis vector.
Params: Initial hypothesis vector, DataSet class, toggle for output type.
Return: If toggle True: Number of time steps to converge
If toggle False: final hypothesis.
"""


def pla(w, data_set, toggle="time"):
    t_converge = 0
    check = True
    while check:
        check = False
        for index in range(data_set.size):
            if not data_set.check(index, w):
                w = update(w, data_set, index)
                check = True
                t_converge += 1
    if toggle == "time":
        return t_converge
    elif toggle == "vector":
        return w
    else:
        return t_converge, w


"""
Method to determine the average number of time step needed for PLA to converge to a valid hypothesis.
Params:  Number of points in data set
Return:  The average number of time steps needed for PLA to converge to valid hypothesis
"""


def convergence_stats(number):
    t_average = 0.0
    e_average = 0.0
    data_set = DataSet(number)
    for i in range(1, 1000):
        data_set.new_set()
        w = np.array([0.0, 0.0, 0.0])
        temp, w = pla(w, data_set, "both")
        t_average = (t_average * i + temp) / (i + 1)
        e_average = (e_average * i + error(data_set, w)) / (i + 1)
    return t_average, e_average



"""
Takes a given hypothesis and target function and calculated the error defined as the probability that any point in the
defined plane will be misclassified under the given hypothesis.
Params:  target function slope and intercept, g slope and intercept.
Return: Probability of miscalculation.
"""


def error(data_set, g):
    error = 0.0
    points = np.random.uniform(-1, 1, (10000, 3))
    for point in points:
        point[2] = 1
        if not data_set.compare(point, g):
            error += 1
    return error / 10000

data_set = DataSet(10)
w = pla(np.array([0.0, 0.0, 0.0]), data_set, "vector")
data_set.visualize_hypoth(w)

