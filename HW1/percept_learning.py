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
    m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
    b = point_a[1] - m*point_a[0]
    return m, b

"""
!!!!Function replaced by vector classify!!!!!
Function to classify points as either correct under current model or incorrect.
Takes an input point as an array as input, with m and b from the current hypothesis.
Returns a boolean.  False if below threshold under give m and b. True if above threshold.
"""
def classify_point(point, m, b):
    target_y = point[0] * m + b
    if (target_y > point[1]):
        return True
    else:
        return False

"""
Function to classify points as either correct under current model or incorrect.
Takes an input point 2D as an array as input, with w vector (in 3D) from the current hypothesis.
Returns a boolean based on dot product of w and point (with 1 added to make array 3D).
False if below or at threshold under give hypothesis. True if above threshold.
"""
def vector_classify(point, w):
    dot = np.dot(np.array([point[0], point[1], 1.0]), w)
    if (dot > 0):
        return True
    else:
        return False

"""
Generates the data set to be used with the PLA.
Takes in the number of points to generate as an argument as well as m, b of target function.
Returns the points as an array of vectors and a boolean array to classify the points compared with a give m, b
"""
def generate_set(number, m, b):
    vectors = np.random.uniform(-1, 1, (number, 2))
    bools = np.ones(number, dtype=bool)
    for count, vector in enumerate(vectors):
        bools[count] = vector_classify(vector, np.array([m, - 1, b]))
    return vectors, bools

"""
Takes the points generated in the generate set function and draws a scatter containing those plots.  The scatter
also shows the target function.  This is used as a visible conformation of the boolean assignation in generate_set()
Takes vectors from generate set and m, b for the target function as params.
Return 1.
"""
def plot_points(points, bools, m, b):
    plt.plot([((-1-b)/m), ((1-b)/m)], [-1, 1])
    for count, point in enumerate(points):
        if (bools[count]):
            plt.plot(point[0], point[1], 'bo')
        else:
            plt.plot(point[0], point[1], 'ro')
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    plt.show()

"""
Takes two vectors as input.  The w vector or hypothesis vector and the misclassified point vector.
Updated the w vector by taking the vector sum of w + x to give an updated w vector.
This updated w vector is the return value.  This w results in the input vector now being correctly classified wrt hypothesis
and target function.
"""
def update(w, x):
    return np.add(w, [1.0, x[0], x[1]])



m, b = target_function()
vectors, bools = generate_set(10, m, b)
# print(vectors)
# print(bools)
# print(m, b)
plot_points(vectors, bools, m, b)
# print(update([1.0, 2.0, 3.0], vectors[0]))