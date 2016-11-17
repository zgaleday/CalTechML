"""
A class to run soft-margin SVM on the features data set included (from postal service data set of zip codes).
Dependencies:  scikit learn and numpy
"""
import numpy as np
from sklearn import svm


class NumberSVM:

    """
    Intakes and cleans the data from features.train for training using various user defined SVM parameters
    and Kernal methods.
    """

    def __init__(self):
        """
        Defines instance variable for use in later methods
        """
        self.X = np.empty(0)
        self.numbers = np.empty(0)
        self.Y = np.empty(0)
        self.svm = None
        self.text_X = np.empty(0)
        self.test_numbers = np.empty(0)
        self.test_Y = np.empty(0)

    def read_data(self, filename, test=False):
        """
        Reads in data points from filename to instance arrays.  If test is false reads into self.X and self.numbers else
        reads into self.test_X and self.test_numbers.  Data is formatted with number in first col.  With features in
        next two, no header.
        :param filename: File in which data is located
        :param test: toggle to select whether set being read in is training or test test. If true test, else training
        :return: void
        """
        file = open(filename, 'r')
        points = []
        numbers = []
        for number, line in enumerate(file):
            line = line.split()
            point = []
            for entry in line[1:]:
                point.append(np.double(entry))
            points.append(point)
            numbers.append([np.double(line[0])])

        file.close()
        self.X = np.array(points, dtype='d')
        self.numbers = np.array(numbers, dtype='d')

    def number_v_all(self, number, test=False):
        """
        Sets the target vector (either self.Y or self.test_Y).  Assigns +1 to any Y[i] where self.numbers[i] == number
        and -1 otherwise.
        :param number: number to set classification to 1 for
        :param test: toggle for training or in sample (true for setting test_Y false for Y)
        :return: void
        """
        # TODO

    def number_v_number(self, a, b, test=False):
        """
        Sets Y[i] = +1 if  numbers[i] == a, Y[i] == -1 if numbers[i] ==b and 0 otherwise
        :param a: number to set classification to +1 for
        :param b: number to set classification to -1 for
        :param test: see methods above
        :return: void
        """
        # TODO

    def poly_solver(self, Q, C):
        """
        Method for solving the polynomial kernel soft-margin SVM problem
        :param Q: Order of the polynomial kernel to be used
        :param C: Margin violation constraint
        :return: void
        """
        # TODO

    def poly_cross_validation(self, Q, C):
        """
        Runs SVM with 10-fold cross validation
        :param Q: Order of the polynomial kernel to be used
        :param C: Margin violation constraint
        :return: Error of the cross validation
        """
        # TODO

    def error_in(self):
        """
        Method for determining the in sample error under the current SVM instance. If the current SVM instance is None
        returns -1
        :return: Error in measure if SVM exists -1 otherwise
        """
        # TODO

    def error_out(self):
        """
        Method for determining the out-of-sample error under the current SVM instance. If the current SVM instance is
        None returns -1
        :return: Out-of-sample error measure if SVM exists -1 otherwise
        """
        # TODO


