__author__ = 'zachary'
import random
import numpy as np
import matplotlib.pyplot as plt

"""
Function generates a random line by choosing two points uniformly ([-1,1], [-1,1]) at random in 2D plane.
The function returns the value of the slope and the value of the intercept to caller.
"""
def target_function():

    point_a = (random.uniform(-1, 1), random.uniform(-1, 1))
    point_b = (random.uniform(-1, 1), random.uniform(-1, 1))
    print(point_a)
    print(point_b)
    m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
    b = point_a[1] - m*point_a[0]
    return m, b

"""
Function to classify points as either correct under current model or incorrect.
Takes an input vector as an array as input, with m and b from the current hypothesis.
Returns a boolean.  False if below threshold under give m and b. True if above threshold.
"""
def classify_points(point, m, b):
    # TODO: Method to classify point as fitting with or against current hypothesis.
    pass

"""
Generates the data set to be used with the PLA.
Takes in the number of points to generate as an argument as well as m, b of target function.
Returns the points as an array of vectors and a boolean array to classify the points compared with a give m, b
"""
def generate_set(number, m, b):
    vectors = np.random.uniform(-1, 1, (number, 2))
    bools = np.ones(number, dtype=bool)
    count = 0
    for vector in vectors:
        x = vector[0]
        target_y = x * m + b
        if (target_y > vector[1]):
            bools[count] = True
        else:
            bools[count] = False
        count += 1

    return vectors, bools

"""
Takes the points generated in the generate set function and draws a scatter containing those plots.  The scatter
also shows the target function.  This is used as a visible conformation of the boolean assignation in generate_set()
Takes vectors from generate set and m, b for the target function as params.
Return 1.
"""
def plot_points(points, bools, m, b):
    count = 0
    plt.plot([((-1-b)/m), ((1-b)/m)], [-1, 1])
    for point in points:
        if (bools[count]):
            plt.plot(point[0], point[1], 'bo')
        else:
            plt.plot(point[0], point[1], 'ro')
        count += 1
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    plt.show()


m, b = target_function()
vectors, bools = generate_set(10, m, b)
print(vectors)
print(bools)
print(m, b)
plot_points(vectors, bools, m, b)