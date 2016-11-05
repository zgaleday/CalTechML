"""
Functions to read in the data set and preform linear regression for classification on a 2D input with the transform
(x1, x2) => (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
Error on classification is defined as the fraction of misclassified points.
"""

def read_file(filename):

    """
    Function to read in the data in a given file.  Format of input is x1 x2 y.
    Data is space delimited and assumes no head.
    :param filename: the file to read data from
    :return: array of points x1, x2 and array of classifications yi
    """
    # TODO: Implement file reading functionality
    pass


def transform(points):

    """
    Takes in an array of points formatted (x1, x2) and outputs an array of points resultant from non-linear transform
    (x1, x2) => (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
    :param points: array of points (x1, x2)
    :return: array of points (1, x1, x2, x1^2, x2^2, x1x2, abs(x1 - x2), abs(x1 + x2))
    """
    # TODO: Implement transform function
    pass


def regression_for_classification(points, classifications):

    """
    Performs linear regression for classification of a set of points given the vector of classifications.
    Returns the hypothesis vector.
    :param points: array of points formatted such that last value in a point in the classification
    :param classifications: vector of classifications {-1, +1}
    :return: hypothesis vector
    """
    # TODO: Implement linear regression for classification
    pass


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
    # TODO: Implement function to determine classification error in a set of points give hypothesis g
    pass


def classify_point(point, classification, g):

    """
    Function to classify a point under a given hypothesis. Out put is {-1, +1}
    :param point: point to classify (array of floats)
    :param classification: +1 or -1 classification
    :param g: hypothesis to classify under
    :return: {-1, +1}
    """
    # TODO: Implement function to classify point under given hypothesis
    pass


