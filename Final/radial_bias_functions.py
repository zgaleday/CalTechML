import numpy as np
from scipy.spatial import distance
from sklearn import svm

"""
Class to compare the radial bias function with SVM using the RBF Kernel for the target function x2 - x1 + 0.25sin(pi *x)
"""

class RadialBiasFunction:

    def __init__(self):

        self.X = np.random.uniform(-1, 1, (100, 2))
        self.Y = np.empty(100)
        self.generate_y()
        self.phi = None
        self.K = 0
        self.centers = None
        self.cluster_sizes = None
        self.g = None
        self.b = None
        self.gamma = None
        self.test_X = np.random.uniform(-1, 1, (10000, 2))
        self.test_Y = np.empty(10000)
        self.generate_y(test=True)

    def classify(self, point):

        """
        Classifies the given point {x1, x2} based on sign x2 - x1 + 0.25sin(pi*x).
        :param point: point formatted {x1, x2}
        :return: +1 if x2 - x1 + 0.25sin(pi * x) pos. -1 otherwise
        """
        x1 = point[0]
        x2 = point[1]
        f_x = x2 - x1 + 0.25 * np.sin(np.pi * x1)
        if f_x > 0:
            return 1.0
        else:
            return -1.0

    def generate_y(self, test=False):

        """
        Generates the Y vector from points X.
        :param test: If true classifies for test set. else classifies for in sample
        :return: void
        """
        if not test:
            for i, point in enumerate(self.X):
                self.Y[i] = self.classify(point)
        else:
            for i, point in enumerate(self.test_X):
                self.test_Y[i] = self.classify(point)

    def cluster(self):

        """
        Preforms K means clustering on the self.X points.  Starts from a random point and runs until convergence.
        If any cluster is empty the algorithm repeats with new starting points.
        Sets self.centers equal to the clusters x1, x2 coords
        :param k: number of clusters
        :return: void
        """
        center_index = np.random.choice(range(100), self.K, replace=False)
        self.centers = np.array([self.X[i] for i in center_index])
        self.cluster_sizes = np.zeros(self.K)
        member_of = np.zeros(100, dtype=int)
        min_dist = np.array([distance.euclidean(self.centers[0], point) for point in self.X])
        self.cluster_sizes[0] = 100
        flag = True
        while flag:
            flag = False
            for i, point in enumerate(self.X):
                for j, center in enumerate(self.centers):
                    if member_of[i] != j:
                        dist = distance.euclidean(point, center)
                        if dist < min_dist[i]:
                            flag = True
                            current = member_of[i]
                            self.cluster_sizes[current] -= 1
                            self.cluster_sizes[j] += 1
                            member_of[i] = j
                            min_dist[i] = dist
            if np.count_nonzero(self.cluster_sizes) != self.K:
                return self.cluster()
            self.centers = np.zeros((self.K, 2), dtype='d')
            for i, point in enumerate(self.X):
                center = member_of[i]
                self.centers[center] += point
            for i, center in enumerate(self.centers):
                center /= self.cluster_sizes[i]

    def generate_phi(self):

        """
        Generates the phi matrix for RBF solving using linear regression for classification.  Stores the value of phi
        in self.phi (100 x K) matrix.  Clustering done in method to ensure dependency met.
        :return: void
        """
        self.phi = np.empty((100, self.K))
        for i, point in enumerate(self.X):
            for j, center in enumerate(self.centers):
                self.phi[i][j] = np.exp(-self.gamma * distance.euclidean(point, center) ** 2)
        self.phi = np.concatenate((self.phi, np.ones((100, 1))), axis=1)

    def LRC(self):

        """
        Does linear regression for classification with phi matrix.  Stores value in self.g_LRC
        :return: void
        """
        pseudo_inverse = np.linalg.pinv(self.phi)
        self.g = np.dot(pseudo_inverse, self.Y)
        self.b = self.g[-1]
        self.g = self.g[:-1]

    def fit(self, gamma, K):
        """
        Solves the vanilla rbf with current in_sample and given params
        :param gamma: gamma value
        :param K: number of clusters
        :return: void
        """
        self.K = K
        self.gamma = gamma
        self.cluster()
        self.generate_phi()
        self.LRC()

    def rbf_classify(self, point):

        """
        Classifies the given point using the calculated rbf classifier
        :param point: point to be classified
        :return: +1 if classifier positive -1 otherwise
        """
        sum = self.b
        for i, center in enumerate(self.centers):
            sum += self.g[i] * np.exp(-self.gamma * distance.euclidean(center, point) ** 2)
        if sum > 0:
            return 1.0
        else:
            return -1.0


    def error(self, in_sample=True):

        """
        Returns the error of the given hypothesis.
        :param in_sample: if true gives in sample. else give out of sample
        :return: error of classification
        """
        if in_sample:
            error = 0.0
            for i, point in enumerate(self.X):
                if self.Y[i] != self.rbf_classify(point):
                    error += 1
            return error / 100
        else:
            error = 0.0
            for i, point in enumerate(self.test_X):
                if self.test_Y[i] != self.rbf_classify(point):
                    error += 1
            return error / 10000

    def resample(self):

        """
        Generates a new in sample set to train on.  Clears all parameters.
        :return: void
        """
        self.X = np.random.uniform(-1, 1, (100, 2))
        self.generate_y()
        self.phi = None
        self.K = 0
        self.centers = None
        self.cluster_sizes = None
        self.g = None
        self.gamma = None


def question_13(gamma):
    """
    Function to determine the number of runs on hard margin svm result in non-zero ein.
    :param gamma: gamma value in rbf kernel
    :return: fraction of times hard-margin w/ svm kernel fails
    """
    rbf = RadialBiasFunction()
    fails = 0.0
    my_svm = svm.SVC(C=np.inf, kernel='rbf', gamma=gamma)
    for i in range(1000):
        my_svm = my_svm.fit(rbf.X, rbf.Y)
        if my_svm.score(rbf.X, rbf.Y) != 1:
            fails += 1
        rbf.resample()
    return fails / 1000


def question_14_and_15(gamma, K):
    """
    Function to determine how often SVM with RBF beats vanilla RBF
    :param gamma: gamma value in rbf kernel
    :param K: number of cluster in regular rbf
    :return: fraction of times hard-margin w/ svm kernel fails
    """
    rbf = RadialBiasFunction()
    wins = 0.0
    my_svm = svm.SVC(C=np.inf, kernel='rbf', gamma=gamma)
    for i in range(100):
        rbf.fit(gamma, K)
        my_svm = my_svm.fit(rbf.X, rbf.Y)
        svm_error = 1 - my_svm.score(rbf.test_X, rbf.test_Y)
        rbf_error = rbf.error(in_sample=False)
        if svm_error < rbf_error:
            wins += 1
        rbf.resample()
    return wins / 100


def question_16():
    """
    Function to compare the performance of regular rbf with 9 clusters and 12 clusters in terms of Eout.
    Print to console for 10 runs
    :return: void
    """
    rbf = RadialBiasFunction()
    for i in range(10):
        rbf.fit(1.5, 9)
        error_in_9 = rbf.error()
        error_out_9 = rbf.error(in_sample=False)
        rbf.fit(1.5, 12)
        error_in_12 = rbf.error()
        error_out_12 = rbf.error(in_sample=False)
        print("Error (in/out) for 9 clusters: {0} / {1}\nError (in/out) for 12 clusters: {2} / {3}\n"
              .format(error_in_9, error_out_9,error_in_12, error_out_12))

question_16()