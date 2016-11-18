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
        self.NVM_X = None
        self.N = -1
        self.svm = None
        self.test_X = None
        self.test_NVN_X = None
        self.test_numbers = None
        self.test_Y = None
        self.index_array = np.empty(11, dtype=int)
        self.NVM_index_array = np.empty(11, dtype=int)
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
            for i, j in enumerate(range(0, np.int(self.N), np.int(self.N / 10))):
                self.index_array[i] = np.int(j)
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
        NVN_X = []
        Y = []
        if not test:
            for i, num in enumerate(self.numbers):
                if num == a:
                    Y.append(np.double(1))
                    NVN_X.append(self.X[i])
                elif num == b:
                    Y.append(np.double(-1))
                    NVN_X.append(self.X[i])
            self.NVM_X = np.array(NVN_X, dtype='d')
            self.Y = np.array(Y, dtype='d')
            n = len(self.NVM_X)
            for i, j in enumerate(range(0, n, np.int(n / 10))):
                self.NVM_index_array[i] = np.int(j)
        else:
            for i, num in enumerate(self.test_numbers):
                if num == a:
                    Y.append(np.double(1))
                    NVN_X.append(self.test_X[i])
                elif num == b:
                    Y.append(np.double(-1))
                    NVN_X.append(self.test_X[i])
            self.test_NVN_X = np.array(NVN_X, dtype='d')
            self.test_Y = np.array(Y, dtype='d')


    def shuffle_arrays(self, ova=True):
        """
        Shuffles the self.X., self.Y, and self.numbers arrays.  The shuffle will occur such that self.X[i], self.Y[i]
        and self.numbers[i] corresponds to the same entry in the input set before the shuffle
        :return: void
        """
        state = np.random.get_state()
        if ova:
            np.random.shuffle(self.X)
            np.random.set_state(state)
            np.random.shuffle(self.Y)
            np.random.set_state(state)
            np.random.shuffle(self.numbers)
        else:
            np.random.shuffle(self.NVM_X)
            np.random.set_state(state)
            np.random.shuffle(self.Y)

    def set_poly_svm_params(self, Q, C):
        """
        Method to set the instance of svm stored in class to poly kernel of degree Q and margin C
        :param Q: Order of the polynomial kernel to be used
        :param C: Margin violation constraint
        :return: void
        """
        self.svm = svm.SVC(C=C, kernel='poly', degree=Q, gamma=1, coef0=1)

    def set_rbf_svm(self, C):
        """
        Method to set the instance of svm stored in class to rbf kernel
        :param C: Margin violation constraint
        """
        self.svm = svm.SVC(C=C, kernel='rbf', gamma=1)

    def svm_solver(self, ova=True):
        """
        Solves current svm instance stored in the class this the self.X and self.Y params.
        If params not set method or svm fails returns false otherwise returns true
        :param ova: If true runs ova solver else runs ovo solver.
        :return: boolean of success of training
        """
        if ova:
            try:
                self.svm.fit(self.X, self.Y)
                return True
            except:
                return False
        else:
            try:
                self.svm.fit(self.NVM_X, self.Y)
                return True
            except:
                return False

    def poly_cross_validation(self, ova=True):
        """
        Runs SVM with 10-fold cross validation
        :param ova: runs ova fit if true ovo fit if false
        :return: Error of the cross validation
        """
        e_cv = 0.0
        if ova:
            temp_X = self.X[self.index_array[1]:]
            temp_Y = self.Y[self.index_array[1]:]
            self.svm.fit(temp_X, temp_Y)
            e_cv += 1 - self.svm.score(self.X[:self.index_array[1]], self.Y[:self.index_array[1]])
            for i in range(1, 10):
                temp_X = np.append(self.X[0: self.index_array[i], 0:2], self.X[self.index_array[i+1]:, 0:2], axis=0)
                temp_Y = np.append(self.Y[0: self.index_array[i]], self.Y[self.index_array[i+1]:])
                self.svm.fit(temp_X, temp_Y)
                e_cv += 1 - self.svm.score(self.X[self.index_array[i]:self.index_array[i + 1]],
                                           self.Y[self.index_array[i]:self.index_array[i + 1]])
            temp_X = self.X[:self.index_array[9]]
            temp_Y = self.Y[:self.index_array[9]]
            self.svm.fit(temp_X, temp_Y)
            e_cv += 1 - self.svm.score(self.X[self.index_array[9]:], self.Y[self.index_array[9]:])
        else:
            temp_X = self.NVM_X[self.NVM_index_array[1]:]
            temp_Y = self.Y[self.NVM_index_array[1]:]
            self.svm.fit(temp_X, temp_Y)
            e_cv += 1 - self.svm.score(self.NVM_X[:self.NVM_index_array[1]], self.Y[:self.NVM_index_array[1]])
            for i in range(1, 10):
                temp_X = np.append(self.NVM_X[0: self.NVM_index_array[i], 0:2],
                                   self.NVM_X[self.NVM_index_array[i + 1]:, 0:2], axis=0)
                temp_Y = np.append(self.Y[0: self.NVM_index_array[i]], self.Y[self.NVM_index_array[i + 1]:])
                self.svm.fit(temp_X, temp_Y)
                e_cv += 1 - self.svm.score(self.NVM_X[self.NVM_index_array[i]:self.NVM_index_array[i + 1]],
                                           self.Y[self.NVM_index_array[i]:self.NVM_index_array[i + 1]])
            temp_X = self.NVM_X[:self.NVM_index_array[9]]
            temp_Y = self.Y[:self.NVM_index_array[9]]
            self.svm.fit(temp_X, temp_Y)
            e_cv += 1 - self.svm.score(self.NVM_X[self.NVM_index_array[9]:], self.Y[self.NVM_index_array[9]:])
        return e_cv / 10


    def error(self, type='in', ova=True):
        """
        Method for determining the in sample error under the current SVM instance. If the current SVM instance is None
        returns -1
        :param type: selecting in or out of sample
        :return: Error in measure if SVM exists -1 otherwise
        """
        if type == 'in':
            try:
                if ova:
                    return 1 - self.svm.score(self.X, self.Y)
                else:
                    return 1 - self.svm.score(self.NVM_X, self.Y)
            except:
                return -1
        elif type == 'out':
            try:
                if ova:
                    return 1 - self.svm.score(self.test_X, self.test_Y)
                else:
                    return 1 - self.svm.score(self.test_NVN_X, self.test_Y)
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
            my_svm.svm_solver(ova=False)
            num_sv = np.sum(my_svm.svm.n_support_)
            ein = my_svm.error(ova=False)
            eout = my_svm.error(type='out', ova=False)
            print("C = {0}, Q ={1},  Number SV = {2}, Ein = {3}, Eout = {4}".format(c, q, num_sv, ein, eout))
            q += 3
        c *= 10


def problem_7():
    """
    Method to determine the C value that most commonly has the lowest E_cv over 100 runs
    """
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.number_v_number(1, 5)
    c_vals = np.array([0.0001, 0.001, 0.01, 0.1, 1.0])
    best_c = np.zeros(5)
    for i in range(100):
        min_c = 10
        min_ecv = 1
        for i, c in enumerate(c_vals):
            my_svm.set_poly_svm_params(2, c)
            e_cv = my_svm.poly_cross_validation(ova=False)
            if e_cv < min_ecv:
                min_c = i
                min_ecv = e_cv
        best_c[min_c] += 1
        my_svm.shuffle_arrays(ova=False)
    for i, c in enumerate(c_vals):
        print("C = {0} was chosen {1} times".format(c, best_c[i]))


def problem_8(trials):
    """
    Methods to solve problems 7 and 8
    """
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.number_v_number(1, 5)

    c = 0.0001
    while c <= 1:
        my_svm.set_poly_svm_params(2, c)
        e_cv = 0
        for i in range(trials):
            e_cv += my_svm.poly_cross_validation(ova=False)
            my_svm.shuffle_arrays(ova=False)

        print("C = {0}, E_cv = {1}".format(c, (e_cv / trials)))
        c *= 10


def problems_9_and_10():
    """
    Method to test num v num svm classification on 1 v 5 with varying C values.  Prints the in sample error to console.
    """
    my_svm = NumberSVM()
    my_svm.read_data("features.train")
    my_svm.read_data("features.test", test=True)
    my_svm.number_v_number(1, 5)
    my_svm.number_v_number(1, 5, test=True)
    c = 0.01
    while c <= 1e6:
        my_svm.set_rbf_svm(c)
        my_svm.svm_solver(ova=False)
        num_sv = np.sum(my_svm.svm.n_support_)
        ein = my_svm.error(ova=False)
        eout = my_svm.error(type='out', ova=False)
        print("C = {0}, Number SV = {1}, Ein = {2}, Eout = {3}".format(c, num_sv, ein, eout))
        c *= 100

problem_7()