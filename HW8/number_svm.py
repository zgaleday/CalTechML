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
        self.X = None
        self.numbers = None
        self.Y = None
        self.N = -1
        self.svm = None
        self.test_X = None
        self.test_numbers = None
        self.test_Y = None
        self.test_points = -1

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
        if not test:
            self.X = np.array(points, dtype='d')
            self.numbers = np.array(numbers, dtype='d')
            self.N = len(self.numbers)
        else:
            self.test_X = np.array(points, dtype='d')
            self.test_numbers = np.array(numbers, dtype='d')
            self.test_points = len(self.test_numbers)

    def number_v_all(self, number, test=False):
        """
        Sets the target vector (either self.Y or self.test_Y).  Assigns +1 to any Y[i] where self.numbers[i] == number
        and -1 otherwise.
        :param number: number to set classification to 1 for
        :param test: toggle for training or in sample (true for setting test_Y false for Y)
        :return: void
        """
        if not test:
            self.Y = np.empty(self.N, dtype='d')
            for i, num in enumerate(self.numbers):
                if num == number:
                    self.Y[i] = 1
                else:
                    self.Y[i] = -1
        else:
            self.test_Y = np.empty(self.test_points, dtype='d')
            for i, num in enumerate(self.test_numbers):
                if num == number:
                    self.test_Y[i] = 1
                else:
                    self.test_Y[i] = -1

    def number_v_number(self, a, b, test=False):
        """
        Sets Y[i] = +1 if  numbers[i] == a, Y[i] == -1 if numbers[i] ==b and 0 otherwise
        :param a: number to set classification to +1 for
        :param b: number to set classification to -1 for
        :param test: see methods above
        :return: void
        """
        if not test:
            self.Y = np.zeros(self.N, dtype='d')
            for i, num in enumerate(self.numbers):
                if num == a:
                    self.Y[i] = 1
                elif num == b:
                    self.Y[i] = -1
        else:
            self.test_Y = np.zeros(self.test_points, dtype='d')
            for i, num in enumerate(self.test_numbers):
                if num == a:
                    self.test_Y[i] = 1
                elif num == b:
                    self.test_Y[i] = -1

    def shuffle_arrays(self):
        """
        Shuffles the self.X., self.Y, and self.numbers arrays.  The shuffle will occur such that self.X[i], self.Y[i]
        and self.numbers[i] corresponds to the same entry in the input set before the shuffle
        :return: void
        """
        state = np.random.get_state()
        np.random.shuffle(self.X)
        np.random.set_state(state)
        np.random.shuffle(self.Y)
        np.random.set_state(state)
        np.random.shuffle(self.numbers)

    def set_poly_svm_params(self, Q, C):
        """
        Method to set the instance of svm stored in class
        :param Q: Order of the polynomial kernel to be used
        :param C: Margin violation constraint
        :return: void
        """
        self.svm = svm.SVC(C=C, kernel='poly', degree=Q)

    def svm_solver(self):
        """
        Solves current svm instance stored in the class this the self.X and self.Y params.
        If params not set method or svm fails returns false otherwise returns true
        :return: boolean of sucess of training
        """
        try:
            self.svm.fit(self.X, self.Y)
            return True
        except:
            return False

    def poly_cross_validation(self, Q, C):
        """
        Runs SVM with 10-fold cross validation
        :param Q: Order of the polynomial kernel to be used
        :param C: Margin violation constraint
        :return: Error of the cross validation
        """
        # TODO

    def error(self, type='in'):
        """
        Method for determining the in sample error under the current SVM instance. If the current SVM instance is None
        returns -1
        :param type: selecting in or out of sample
        :return: Error in measure if SVM exists -1 otherwise
        """
        if type == 'in':
            try:
                return 1 - self.svm.score(self.X, self.Y)
            except:
                return -1
        elif type == 'out':
            try:
                return 1 - self.svm.score(self.test_X, self.test_Y)
            except:
                return -1
        return -1


def problems_2_3():
    """Method to find the error in for all number versus all classifications.
    Prints error in to console:
    """
    num_set = range(10)
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.set_poly_svm_params(2, 0.01)
    for num in num_set:
        my_svm.number_v_all(num)
        my_svm.svm_solver()
        print("In sample error for {0} versus all: ".format(num), my_svm.error())


def problem_4():
    """
    Method to determine the difference in the number of support vectors resultant from one versus all (lowest Ein) and
     zero versus all (highest Ein).
     Prints to console
    """
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.set_poly_svm_params(2, 0.01)
    my_svm.number_v_all(0)
    my_svm.svm_solver()
    zero_support_vectors = np.sum(my_svm.svm.n_support_)
    my_svm.number_v_all(1)
    my_svm.svm_solver()
    print("The difference in the number of support vectors is: ", np.sum(my_svm.svm.n_support_) - zero_support_vectors)


def problem_5_and_6():
    """
    Method to test num v num svm classification on 1 v 5 with varying C values.  Prints the in sample error to console.
    """
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.read_data("features.test", test=True)
    my_svm.number_v_number(1, 5)
    my_svm.number_v_number(1, 5, test=True)
    c = 0.0001
    while c <= 1:
        q = 2
        while q <= 5:
            my_svm.set_poly_svm_params(q, c)
            my_svm.svm_solver()
            num_sv = np.sum(my_svm.svm.n_support_)
            ein = my_svm.error()
            eout = my_svm.error(type='out')
            print("C = {0}, Q ={1}  Number SV = {2}, Ein = {3}, Eout = {4}".format(c, q, num_sv, ein, eout))
            q += 3
        c *= 10


