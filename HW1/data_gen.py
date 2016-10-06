import random
import numpy as np
import matplotlib.pyplot as plt


class DataSet:
    """Class to generate and manipulate a uniform distribution in 2D with a defined target function
    (randomly generated)"""

    def __init__(self, size):
        self.size = size
        self.bools = np.ones(self.size, dtype=bool)
        self.points = np.random.uniform(-1, 1, (self.size, 3))
        self.target = [0, 0, 0]
        self.m = 0
        self.b = 0
        self.target_function()
        self.generate_set()

    """
    Function generates a random line by choosing two points uniformly ([-1,1], [-1,1]) at random in 2D plane.
    The function returns the value of the slope and the value of the intercept to caller.
    """

    def target_function(self):
        point_a = (random.uniform(-1, 1), random.uniform(-1, 1))
        point_b = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
        self.b = point_a[1] - self.m * point_a[0]
        self.target = np.array([self.m, -1, self.b])

    """
    Generates the data set.
    Takes in the number of points to generate as an argument as well as m, b of target function.
    Returns the points as an array of vectors and a boolean array to classify the points compared with a give m, b
    """

    def generate_set(self):
        for count, point in enumerate(self.points):
            point[2] = 1.0
            self.classify(point, count)


    """Classified the points compared to the target function.
    If dot is positive then classified as True or +1. If dot negative
    classify as -1.
    """

    def classify(self, point, index):
        dot = np.dot(point, self.target)
        if dot > 0:
            self.bools[index] = True
        else:
            self.bools[index] = False


    """Check if classification matches for target and given input vector
    Param: vector and index of point
    Return: True if classification matches false otherwise
    """
    def check(self, index, h):
        dot = np.dot(self.points[index], h)
        if (dot > 0 and self.bools[index]) or (dot <= 0 and self.bools[index] == False):
            return True
        else:
            return False


    """
    Takes the points generated in the generate set function and draws a scatter containing those plots.  The scatter
    also shows the target function.  This is used as a visible conformation of the boolean assignation in generate_set()
    Takes vectors from generate set and m, b for the target function as params.
    Return 1.
    """

    def plot_points(self):
        plt.plot([((-1 - self.b) / self.m), ((1 - self.b) / self.m)], [-1, 1], 'r')
        for count, point in enumerate(self.points):
            if self.bools[count]:
                plt.plot(point[0], point[1], 'bo')
            else:
                plt.plot(point[0], point[1], 'ro')
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        plt.show()


dist = DataSet(100)
dist.plot_points()