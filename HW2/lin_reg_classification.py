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


def linear_regression(data_set):

    """
    A method to computed the least squares linear regression for classification.

    Params: DataSet object
    Return: g calculated using linear regression.
    """
    target = generate_target(data_set)
    if data_set.linear:
        pseudo_inverse = np.linalg.pinv(data_set.points)
        w = np.dot(pseudo_inverse, target)
    else:
        pseudo_inverse = np.linalg.pinv(data_set.transform)
        w = np.dot(pseudo_inverse, target)

    return w


def error_in(data_set, g):

    """Method to determine the in sample error of the linear regression classification method.

    Params: DataSet object, and the vector for output from linear_regression method
    Return: The in sample error probability
    """
    error = 0.0
    for index, point in enumerate(data_set.points):
        if data_set.linear:
            if not data_set.compare(point, g):
                error += 1
        else:
            if not data_set.check(index, g):
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
    if data_set.linear:
        for point in points:
            point[2] = 1
            if not data_set.compare(point, g):
                error += 1
    else:
        transform = np.empty((1000, 6))
        data_set.do_transform(transform, points)
        for vector in transform:
            dot = np.dot(vector, g)
            if ((dot > 0 and data_set.nonlinear_classify(vector) == False)
                or (dot <= 0 and data_set.nonlinear_classify(vector) == True)):
                error += 1
    return error / 1000


def average_error(number, type="in", linear=True, threshold=0.0, noise=0.0):

    """Method to calculate the in sample or out of sample error average over 1000 runs

    Params: Number of points in data set to be generated, type (valid types are "in" and "out"
    Return: the average error over 1000 runs as a float
    """
    error = 0.0
    data_set = DataSet(number, linear=linear, threshold=threshold, noise=noise)
    for i in range(1000):
        data_set.new_set()
        if type == "in":
            temp = error_in(data_set, linear_regression(data_set))
            error = (error * i + temp) / (i + 1)
        elif type == "out":
            temp = error_out(data_set, linear_regression(data_set))
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
        w = linear_regression(data_set)
        temp = pla.pla(w, data_set)
        t_average = (t_average * i + temp) / (i + 1)
    return t_average



# print(average_error(1000, type="out", linear=False, threshold=0.6, noise = 0.1))