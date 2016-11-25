from HW8.number_svm import NumberSVM
import HW6.reg_weight_decay as LRC
import numpy as np

def problems_7_through_9(transform=False):
    """
    Method to solve HW problems 7-9.  Does a one versus all LRWWD for all numbers and prints out the Ein and Eout
    to console.
    :param transform: determines weather a nonlinear transform is done for the data.
    :return: void
    """
    data = NumberSVM()
    data.read_data("features.train")
    data.read_data("features.test", test=True)
    for number in range(10):
        data.number_v_all(number)
        data.number_v_all(number, test=True)
        if transform:
            transformed_X = LRC.transform(data.X)
            transformed_test_X = LRC.transform(data.test_X)
            g = LRC.weight_decay_lr_classification(transformed_X, data.Y, 0)
            ein = LRC.classification_error(transformed_X, data.Y, g)
            eout = LRC.classification_error(transformed_test_X, data.test_Y, g)
        else:
            g = LRC.weight_decay_lr_classification(data.X, data.Y, 0)
            ein = LRC.classification_error(data.X, data.Y, g)
            eout = LRC.classification_error(data.test_X, data.test_Y, g)
        print("Number: {0}, Ein: {1}, Eout: {2}".format(number, ein, eout))


def problem_10():
    """
    Method to solve problem 10.  Runs 1 v 5 classifier with non-linear transform and compares the preformance of lambda
    {.01, 1} by printing Ein and Eout to console
    :return: Void
    """
    data = NumberSVM()
    data.read_data("features.train")
    data.read_data("features.test", test=True)
    data.number_v_number(1, 5)
    data.number_v_number(1, 5, test=True)
    data.NVM_X = LRC.transform_standard(data.NVM_X)
    data.test_NVN_X = LRC.transform_standard(data.test_NVN_X)
    g_low = LRC.weight_decay_lr_classification(data.NVM_X, data.Y, -2)
    g_high = LRC.weight_decay_lr_classification(data.NVM_X, data.Y, 0)
    ein_low = LRC.classification_error(data.NVM_X, data.Y, g_low)
    eout_low = LRC.classification_error(data.test_NVN_X, data.test_Y, g_low)
    ein_high = LRC.classification_error(data.NVM_X, data.Y, g_high)
    eout_high = LRC.classification_error(data.test_NVN_X, data.test_Y, g_high)
    print("lamda = 0.01: Ein: {0}, Eout: {1} \nlambda = 1.00: Ein: {2}, Eout: {3}".format(ein_low, eout_low,
                                                                                          ein_high, eout_high))


problem_10()