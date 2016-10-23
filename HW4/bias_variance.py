import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def linear_regression(points):

    """
    A method to computed the least squares linear regression for classification for ax target.

    Params: two sets of x,y coords in a 1x4 array
    Return: a calculated by LR.
    """
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x = np.array([[x1], [x2]])
    target = ([[y1], [y2]])
    pseudo_inverse = np.linalg.pinv(x)
    a = np.dot(pseudo_inverse, target)
    # plt.plot([-1, a], [1, a])
    # plt.plot(x1, y1 , 'ro')
    # plt.plot(x2, y2, 'ro')
    # plt.axis([-1, 1, -1, 1])
    # plt.show()

    return a


def repeats(trials):

    """
    Runs linear_regression trials times recording the a values and returing an array of the values of lenght trials
    :param trials: number of trials
    :return: array of trials length with a values
    """
    result = np.empty(trials)
    for trial in range(trials):
        temp = np.random.uniform(-1, 1, 4)
        temp[1] = np.sin(np.pi * temp[0])
        temp[3] = np.sin(np.pi * temp[2])
        result[trial] = linear_regression(temp)

    return result

def var(result, avg):

    """
    Calculates the variance of the ax solution
    :param result: array of a's from repeats
    :param avg: average hypothesis
    :return: the expected value of the variance
    """
    expected = np.empty(result.size)
    for count, a in enumerate(result):
        temp_var = 0.5 *integrate.quad((lambda x: ((avg * x - a * x) ** 2)), -1, 1)[0]
        expected[count] = temp_var
        # print(a, temp_var)
    return np.average(expected)

def bias(avg):
    return 0.5 * integrate.quad((lambda x: ((avg * x - np.sin(np.pi * x)) ** 2)), -1, 1)[0]

a = repeats(100000)
average = np.average(a)
variance = var(a, average)
print("Average: ", average)
print("var: ", var(a, average))
print("bias: ", bias(average))
