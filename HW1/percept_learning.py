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
    m = (point_a[1] - point_b[1]) / point_a[0] - point_b[0]
    b = point_a[1] - m*point_a[0]
    return m, b

"""
Function to classify points as either correct under current model or incorrect.
Takes an input vector as an array as input, with w from the current hypothesis, and a boolean array and index into the array.
Returns a boolean.  False if misclassified under give m and b. True is properly classified.
"""
def classify_points():
    TODO

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
def plot_points():
    TODO





m, b = target_function()
vectors, bools = generate_set(10, m, b)
print(vectors)
print(bools)
print(m, b)
