"""
Functions to read in the data set and preform linear regression for classification on a 2D input with the transform
(x1, x2) => (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
Error on classification is defined as the fraction of misclassified points.
"""
import numpy as np


def read_file(filename):

    """
    Function to read in the data in a given file.  Format of input is x1 x2 y.
    Data is space delimited and assumes no head.
    :param filename: the file to read data from
    :return: array of points x1, x2 and array of classifications yi
    """
    points = []
    classification = []
    file = open(filename, 'r')
    for number, line in enumerate(file):
        line = line.split()
        point = []
        for entry in line[:-1]:
            point.append(float(entry))
        points.append(point)
        classification.append([float(line[-1])])

    file.close()
    return np.array(points), np.array(classification)


def transform(points):

    """
    Takes in an array of points formatted (x1, x2) and outputs an array of points resultant from non-linear transform
    (x1, x2) => (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
    :param points: array of points (x1, x2)
    :return: array of points (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
    """
    transformed = np.empty([len(points), 8])
    for index, point in enumerate(points):
        x1 = point[0]
        x2 = point[1]
        transformed[index][0] = 1.0
        transformed[index][1] = x1
        transformed[index][2] = x2
        transformed[index][3] = x1 ** 2
        transformed[index][4] = x2 ** 2
        transformed[index][5] = x1 * x2
        transformed[index][6] = abs(x1 - x2)
        transformed[index][7] = abs(x1 + x2)
    return transformed


def regression_for_classification(points, classifications):

    """
    Performs linear regression for classification of a set of points given the vector of classifications.
    Returns the hypothesis vector.
    :param points: array of points formatted such that last value in a point in the classification
    :param classifications: vector of classifications {-1, +1}
    :return: hypothesis vector
    """
    pseudo_inverse = np.linalg.pinv(points)
    w = np.dot(pseudo_inverse, classifications)
    return w


def weight_decay_lr_classification(points, classifications, k):

    """
    Performs LR for classifcation with weight decay regularizations with lambda = 10^k
    :param points: array of points formatted such that last value in a point in the classification
    :param classifications: vector of classifications {-1, +1}
    :param k: value of exponent for lambda
    :return: hypothesis vector resultant from LR with weight decay
    """
    # TODO: Implement LR for classification with weight decay
    pass


def classification_error(points, classifications, g):

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
        if classifications[index] != classify_point(point, g):
            misclassified += 1

    return misclassified / len(points)


def classify_point(point, g):

    """
    Function to classify a point under a given hypothesis. Out put is {-1, +1}
    :param point: point to classify (array of floats)
    :param g: hypothesis to classify under
    :return: {-1, +1}
    """
    sign = np.sign(np.dot(point, g))
    if sign > 0:
        return 1.0
    else:
        return -1.0


points, classifications = read_file("in.dta")
points_out, classifications_out = read_file("out.dta")
points = transform(points)
points_out = transform(points_out)
g = regression_for_classification(points, classifications)
print(classification_error(points, classifications, g))
print(classification_error(points_out, classifications_out, g))
