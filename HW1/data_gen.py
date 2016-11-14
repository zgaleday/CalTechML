import numpy as np
import matplotlib.pyplot as plt


class DataSet:
    """Class to generate and manipulate a uniform distribution in 2D with a defined target function
    (randomly generated)"""

    def __init__(self, size, linear=True, threshold=0, noise=0, generate=True):
        self.size = size
        self.bools = np.ones(self.size, dtype=bool)
        self.points = np.random.uniform(-1, 1, (self.size, 3))
        self.transform = np.empty((self.size, 6))
        self.target = [0, 0, 0]
        self.m = 0
        self.b = 0
        self.linear = linear
        self.threshold = threshold
        self.noise = noise
        self.both_sides = False
        self.support_vector = np.zeros(self.size, dtype=bool)
        if linear:
            self.target_function()
        else:
            self.do_transform(self.transform, self.points)
        if generate:
            self.generate_set()



    """Generates a new set of points and target function for the DataSet class"""


    def new_set(self):
        if self.linear:
            self.target_function()
        self.support_vector = np.zeros(self.size, dtype=bool)
        for count, point in enumerate(self.points):
            point[0] = np.random.uniform(-1, 1)
            point[1] = np.random.uniform(-1, 1)
            if self.linear:
                self.classify(point, count)
            else:
                self.do_transform(self.transform, self.points)
                self.bools[count] = self.nonlinear_classify(point)

    """
    Function generates a random line by choosing two points uniformly ([-1,1], [-1,1]) at random in 2D plane.
    The function returns the value of the slope and the value of the intercept to caller.
    """

    def target_function(self):
        point_a = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        point_b = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        self.m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
        self.b = point_a[1] - self.m * point_a[0]
        self.target = np.array([self.m, -1, self.b])


    """
    Function that generated the transform x, y, xy, x^2, y^2, 1 of the generate set of points.

    Params: none
    Return: none
    """
    def do_transform(self, transform, array):
        for index,point in enumerate(array):
            transform[index][0] = point[0]
            transform[index][1] = point[1]
            transform[index][2] = point[1] * point[2]
            transform[index][3] = point[0] ** 2
            transform[index][4] = point[1] ** 2
            transform[index][5] = 1




    """Picks what set to generate based on linear boolean in class init.
    Generates linear if LINEAR is True non-linear otherwise.
    """

    def generate_set(self):
        if self.linear:
            self.linear_generate_set()
        else:
            self.quad_generate_set()

    """
    Generates the data set.
    See classify method for details.
    If all points on one side of target generates a new set
    Return: none
    """

    def linear_generate_set(self):
        both = None
        for count, point in enumerate(self.points):
            if count == 0:
                both = self.classify(point, count)
                point[2] = 1.0
            else:
                point[2] = 1.0
                if both != self.classify(point, count):
                    self.both_sides = True
        if not self.both_sides:
            self.new_set()


    """
    Generates a non-linearly classified data set.
    See non-linear classify for details.
    Params:
    """
    def quad_generate_set(self):
        for count, point in enumerate(self.points):
            point[2] = 1.0
            self.bools[count] = self.nonlinear_classify(point)



    """Classified the points compared to the target function.
    If dot is positive then classified as True or +1. If dot negative
    classify as -1.
    """

    def classify(self, point, index):
        dot = np.dot(point, self.target)
        if dot > 0:
            self.bools[index] = True
            return True
        else:
            self.bools[index] = False
            return False


    """Check if classification matches for target and given input vector
    Param: vector and index of point
    Return: True if classification matches false otherwise
    """
    def check(self, index, h):
        if self.linear:
            dot = np.dot(self.points[index], h)
        else:
            dot = np.dot(self.transform[index], h)

        if (dot > 0 and self.bools[index]) or (dot <= 0 and self.bools[index] == False):
            return True
        else:
            return False

    """
    Compared the classification of given point with against the target function and a given vector
    Params: Point, vector
    Return: True if target and vector class ==, false otherwise
    """
    def compare(self, point, g):
        dot = np.dot(point, self.target)
        dot_g = np.dot(point, g)
        if np.sign(dot) == np.sign(dot_g):
            return True
        else:
            return False

    """
    Takes the points generated in the generate set function and draws a scatter containing those plots.  The scatter
    also shows the target function.  This is used as a visible conformation of the boolean assignation in generate_set()
    Takes vectors from generate set and m, b for the target function as params.
    Return 1.
    """

    def plot_points(self, plot=False):
        plt.plot([((-1 - self.b) / self.m), ((1 - self.b) / self.m)], [-1, 1], 'r')
        for count, point in enumerate(self.points):
            if self.support_vector[count]:
                if self.bools[count]:
                    plt.plot(point[0], point[1], 'bx')
                else:
                    plt.plot(point[0], point[1], 'rx')
            else:
                if self.bools[count]:
                    plt.plot(point[0], point[1], 'bo')
                else:
                    plt.plot(point[0], point[1], 'ro')
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        if plot:
            plt.show()

    """
    Takes the points in the data set the target function and a given vector and plots them all
    Param: vector
    Return: None
    """

    def visualize_hypoth(self, g):
        self.plot_points()
        slope, inter = self.vector_to_standard(g)
        plt.plot([((-1 - inter) / slope), ((1 - inter) / slope)], [-1, 1], 'b')
        plt.show()

    """
    Function to take hypothesis vector and return slope and intercept
    Params: Hypothesis vector
    Return: m, b of hypothesis vector
    """

    def vector_to_standard(self, w):
        m = (- 1 / w[1]) * w[0]
        b = (- 1 / w[1]) * w[2]
        return m, b


    """
    A function classify points according to a non-linear target.  Has a toggle function that either adds or removes noise
    to the target function. Target will be of the form x^2 + y^2 - threshold
    Params: Threshold of quad target (float), point to be classified,
    and noise in the classification (prob of miscalc 0-1).
    Return: True if sign x_1^2 + x_2^2 - threshold positive, False otherwise.
    """

    def nonlinear_classify(self, point):
        temp = np.sign(point[0] ** 2 + point[1] ** 2 - self.threshold)
        det = np.random.random()
        if (temp == 1 and det < self.noise) or (temp != 1 and det > self.noise):
            return True
        else:
            return False
