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
        self.phi = None
        self.K = 0
        self.centers = None
        self.cluster_sizes = None
        self.g = None
        self.gamma = None
        self.test_X = None
        self.test_Y = None

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

    def cluster(self, k):

        """
        Preforms K means clustering on the self.X points.  Starts from a random point and runs until convergence.
        If any cluster is empty the algorithm repeats with new starting points.
        Sets self.centers equal to the clusters x1, x2 coords
        :param k: number of clusters
        :return: void
        """
        self.K = k
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
                return self.cluster(k)
            self.centers = np.zeros((self.K, 2), dtype='d')
            for i, point in enumerate(self.X):
                center = member_of[i]
                self.centers[center] += point
            for i, center in enumerate(self.centers):
                center /= self.cluster_sizes[i]

    def generate_phi(self, gamma):

        """
        Generates the phi matrix for RBF solving using linear regression for classification.  Stores the value of phi
        in self.phi (100 x K) matrix.  Clustering done in method to ensure dependency met.
        :param gamma: the value of gamma in the RBF model
        :return: void
        """
        self.gamma = gamma
        self.phi = np.empty((100, self.K))
        for i, point in enumerate(self.X):
            for j, center in enumerate(self.centers):
                self.phi[i][j] = np.exp(-gamma * distance.euclidean(point, center) ** 2)

    def LRC(self):

        """
        Does linear regression for classification with phi matrix.  Stores value in self.g_LRC
        :return: void
        """
        pseudo_inverse = np.linalg.pinv(self.phi)
        self.g = np.dot(pseudo_inverse, self.Y)

    def rbf_classify(self, point):

        """
        Classifies the given point using the calculated rbf classifier
        :param point: point to be classified
        :return: +1 if classifier positive -1 otherwise
        """
        sum = 0.0
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
            self.test_X = np.random.uniform(-1, 1, (10000, 2))
            self.generate_y(test=True)
            error = 0.0
            for i, point in enumerate(self.test_X):
                if self.test_Y[i] != self.rbf_classify(point):
                    error += 1
            return error / 10000





rbf = RadialBiasFunction()
rbf.generate_y()
rbf.cluster(7)
rbf.generate_phi(1.5)
rbf.LRC()
print(rbf.g)
