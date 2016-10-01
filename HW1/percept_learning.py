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
    plt.plot([((-1-b)/m), ((1-b)/m)], [-1, 1], 'r')
    for count, point in enumerate(points):
        if bools[count]:
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
def update(w, x, bools, index):
    vect = np.array([x[0], x[1], 1.0])
    # dot = np.dot(vect, w)
    if bools[index]:
        d = 1
    else:
        d = -1
    vect = np.multiply(d, vect)
    temp = np.add(w, vect)
    return temp



"""
A method to run the PLA on the generated data set.  Hypothesis starts as the 3D Z vector.  All points misclassified
under this hypoth.  Picks a random misclassified point wrt training value and updates the hypothesis with update().
Runs until all points correctly classified under hypothesis.  Returns the hypothesis vector.
Params: Points of data set (array of 2D arrays), boolean array of value of each point wrt target function,
and vector for first hypothesis. Toggle automatically set to True.
Return: If toggle True: Number of time steps to converge
If toggle False: final hypothesis.
"""
def pla(points, bools, w, toggle="time"):
    t_converge = 0
    while(True):
        check = False
        for count, point in enumerate(points):
            classification = vector_classify(point, w)
            if bools[count] != classification:
                w = update(w, point, bools, count)
                check = True
                t_converge += 1
        if check != True:
            break
    if toggle == "time":
        return t_converge
    elif toggle == "vector":
        return w
    else:
        return t_converge, w

"""
Adds line generated by PLA to the graph to show g line compared to f. Must be called before plot_points.
Params: g vector from PLA
Return: None
"""
def graph_g(g):
    m, b = vector_to_standard(g)
    plt.plot([((-1 - b) / m), ((1 - b) / m)], [-1, 1], 'b')


"""
Method to determine the average number of time step needed for PLA to converge to a valid hypothesis.
Params:  Number of points in data set
Return:  The average number of time steps needed for PLA to converge to valid hypothesis
"""
def convergence_time(number):
    t_average = 0.0
    e_average = 0.0
    for i in range(1,1000):
        m, b = target_function()
        vectors, bools = generate_set(number, m, b)
        w = np.array([0.0, 0.0, 0.0])
        temp, w = pla(vectors, bools, w, "both")
        t_average = (t_average * i + temp) / (i + 1)
        e_average = (e_average * i + error(np.array([m, - 1, b]), w)) / (i + 1)
    return t_average, e_average
"""
Function to take hypothesis vector and return slope and intercept
Params: Hypothesis vector
Return: m, b of hypothesis vector
"""
def vector_to_standard(w):
    m = (- 1 / w[1]) * w[0]
    b = (- 1 / w[1]) * w[2]
    return m, b


"""
Takes a given hypothesis and target function and calculated the error defined as the probability that any point in the
defined plane will be misclassified under the given hypothesis.
Params:  target function slope and intercept, g slope and intercept.
Return: Probability of miscalculation.
"""
def error(f, g):
    error = 0.0
    points = np.random.uniform(-1, 1, (10000, 2))
    for point in points:
        if vector_classify(point, f) != vector_classify(point, g):
            error += 1
    return error / 10000



print(convergence_time(10))
# m, b = target_function()
# vectors, bools = generate_set(100, m, b)
# w = [0.0, 0.0, 0.0]
# g = pla(vectors, bools, w, "vector")
# print(error(np.array([m, - 1, b]), g))
# graph_g(g)
# plot_points(vectors, bools, m, b)