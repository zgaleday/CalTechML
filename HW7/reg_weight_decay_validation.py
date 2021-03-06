"""
Functions to read in the data set and preform linear regression for classification on a 2D input with the transform
(x1, x2) => (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
Error on classification is defined as the fraction of misclassified points.
Implement validation set.
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


def split_set(points):

    """Takes in an array of data points formatted as a 2D array and returns a validation set of the last 10 elements
    and a training set of the previous n-10 elements
    :param points: array of points (x1, x2,......)
    :return: an array of training points and validation points
    """
    return points[:-10], points[-10:]


def regression_for_classification(points, classifications, terms=8):

    """
    Performs linear regression for classification of a set of points given the vector of classifications.
    Returns the hypothesis vector.
    :param points: array of points formatted such that last value in a point in the classification
    :param classifications: vector of classifications {-1, +1}
    :param terms: number of terms to do LR on (default value == 8)
    :return: hypothesis vector
    """
    copy = np.empty([len(points), terms])
    for index, point in enumerate(points):
        copy[index] = point[:terms]
    pseudo_inverse = np.linalg.pinv(copy)
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
    lam = 10 ** k
    z_transposed_z = np.dot(points.transpose(), points)
    inverse = np.linalg.inv(np.add(z_transposed_z, np.multiply(lam, np.identity(len(z_transposed_z)))))
    pseudo_pseudo = np.dot(inverse, points.transpose())
    return np.dot(pseudo_pseudo, classifications)


def classification_error(points, classifications, g, terms=8):

    """
    Determines the classification error under a given hypothesis g on a set of points formatted as above (1, x1, ....)
    with give classification vector
    :param points: points to determine classification error on
    :param classifications: vector of classifications {-1, +1}
    :param g: hypothesis vector
    :param terms: number of terms being used in input set
    :return: fraction of points misclassified
    """
    misclassified = 0.0
    for index, point in enumerate(points):
        if classifications[index] != classify_point(point, g, terms):
            misclassified += 1

    return misclassified / len(points)


def classify_point(point, g, terms=8):

    """
    Function to classify a point under a given hypothesis. Out put is {-1, +1}
    :param point: point to classify (array of floats)
    :param g: hypothesis to classify under
    :param terms: number of terms being used in input set
    :return: {-1, +1}
    """
    sign = np.sign(np.dot(point[:terms], g))
    if sign > 0:
        return 1.0
    else:
        return -1.0


def select_model(training, validation, training_classification, validation_classification):

    """
    Defines a method to select model complexity using training and validation set.
    :param training: training set of points (format as described by transform)
    :param validation: validation set of points (format as described by transform)
    :param trainging_classification: classification vector for training set
    :param validation_classification: classification vector for validation set
    :return: the model complexity with the least validation error, model complexity with least Eout
    """
    min_val_error = 1.1
    min_val_k = -1
    points_out, classifications_out = read_file("out.dta")
    points_out = transform(points_out)
    min_eout = 1.1
    min_eout_k = -1
    for k in range(4, 9):
        w = regression_for_classification(training, training_classification, k)
        error_val = classification_error(validation, validation_classification, w, k)
        error_out = classification_error(points_out, classifications_out, w, k)
        if error_val < min_val_error:
            min_val_error = error_val
            min_val_k = k
        if error_out < min_eout:
            min_eout = error_out
            min_eout_k = k
    return min_val_k - 1, min_eout_k - 1


points, classifications = read_file("in.dta")
points_out, classifications_out = read_file("out.dta")
points = transform(points)
training, validation = split_set(points)
training_class, validation_class = split_set(classifications)
print(select_model(training, validation, training_class, validation_class))
