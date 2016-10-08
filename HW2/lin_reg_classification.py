from HW1.data_gen import DataSet
import numpy as np
import HW1.percept_learning as pla


def generate_target(data_set):

    """A method used to set the target vector for the linear regression.

    Target vector i is set to -1 if point i dot f is negative (or zero) otherwise set to +1.
    Params: DataSet
    Return: Target vector as an numpy array (length == number points in DataSet)
    """
    target_vector = np.empty(data_set.size, dtype=float)
    for index, bool in enumerate(data_set.bools):
        if bool:
            target_vector[index] = 1
        else:
            target_vector[index] = -1

    return target_vector


def linear_regression(data_set, target):

    """
    A method to computed the least squares linear regression for classification.

    Params: DataSet object
    Return: g calculated using linear regression.
    """
    pseudo_inverse = np.linalg.pinv(data_set.points)
    w = np.dot(pseudo_inverse, target)

    return w


def error_in(data_set, g):

    """Method to determine the in sample error of the linear regression classification method.

    Params: DataSet object, and the vector for output from linear_regression method
    Return: The in sample error probability
    """
    error = 0.0
    for point in data_set.points:
        if not data_set.compare(point, g):
            error += 1
    return error / data_set.size


def error_out(data_set, g):

    """Calculated out of sample error of linear regression classification algorithm stocastically.

    Tests 1000 randomly generated points and compares the classification of the LR classification alg and target.
    Params: DataSet object.  Hypothesis vector.
    Return: Stocastically determined out of sample error for LR classification algorithm.
    """
    error = 0.0
    points = np.random.uniform(-1, 1, (1000, 3))
    for point in points:
        point[2] = 1
        if not data_set.compare(point, g):
            error += 1
    return error / 1000


def average_error(number, type="in"):

    """Method to calculate the in sample or out of sample error average over 1000 runs

    Params: Number of points in data set to be generated, type (valid types are "in" and "out"
    Return: the average error over 1000 runs as a float
    """
    error = 0.0
    data_set = DataSet(number)
    for i in range(1000):
        data_set.new_set()
        if type == "in":
            temp = error_in(data_set, linear_regression(data_set, generate_target(data_set)))
            error = (error * i + temp) / (i + 1)
        elif type == "out":
            temp = error_out(data_set, linear_regression(data_set, generate_target(data_set)))
            error = (error * i + temp) / (i + 1)
        else:
            raise ValueError("type must be the string 'in' or 'out'.")
    return error


def convergence_time(number):

    """Use LR as output for start vector of PLA algorithm.  Outputs average time (1000 trials) of pla before convergence

    Params: Number of points in data set
    Return: Average time of convergence for modified PLA
    """
    t_average = 0.0
    data_set = DataSet(number)
    for i in range(1000):
        data_set.new_set()
        w = linear_regression(data_set, generate_target(data_set))
        temp = pla.pla(w, data_set)
        t_average = (t_average * i + temp) / (i + 1)
    return t_average


print(convergence_time(10))