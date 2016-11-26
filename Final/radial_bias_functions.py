import numpy as np

"""
Class to compare the radial bias function with SVM using the RBF Kernel for the target function x2 - x1 + 0.25sin(pi *x)
"""

class RadialBiasFunction:

    def __init__(self):

        self.X = np.random.uniform(-1, 1, (100, 2))
        self.Y = np.array((100))
        self.phi = None
        self.K = 0
        self.centers = None
        self.cluster_sizes = None
        self.g_LRC = None
        self.RBF_svm = None

    def classify(self, point):

        """
        Classifies the given point {x1, x2} based on sign x2 - x1 + 0.25sin(pi*x).
        :param point: point formatted {x1, x2}
        :return: +1 if x2 - x1 + 0.25sin(pi * x) pos. -1 otherwise
        """

    def generate_y(self):

        """
        Generates the Y vector from points X.
        :return: void
        """


    def cluster(self, K):

        """
        Preforms K means clustering on the self.X points.  Starts from a random point and runs until convergence.
        If any cluster is empty the algorithm repeats with new starting points.
        Sets self.centers equal to the clusters x1, x2 coords
        :param K: number of clusters
        :return: void
        """

    def generate_phi(self, gamma):

        """
        Generates the phi matrix for RBF solving using linear regression for classification.  Stores the value of phi
        in self.phi (100 x K) matrix.  Clustering done in method to ensure dependency met.
        :param gamma: the value of gamma in the RBF model
        :return: void
        """

    def RBF(self, point, gamma, center):

        """
        Calculates the RBF for the given point an gamma value with the center at index center in self.center
        :param point: {x1, x2}
        :param gamma: gamma value in RBF
        :param center: index into the self.center array
        :return: value of the RBF for given input.
        """

    def LRC(self):

        """
        Does linear regression for classification with phi matrix.  Stores value in self.g_LRC
        :return: void
        """

    def RBF_SVM(self, gamma):

        """
        Does Hard-margin SVM with RBF kernel and stores the svm instance in self.RBF_svm
        :param gamma: the gamma value in the RBF kernel
        :return: void
        """

    def error(self, in_sample=True):
        """
        Returns the error of the given hypothesis.
        :param in_sample: if true gives in sample. else give out of sample
        :return: error of classification
        """
