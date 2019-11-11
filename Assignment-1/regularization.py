import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# L1 Regularization: Lasso Regression
# L2 Regularization: Ridge Regression

# constants
data = pd.read_csv('../dataset/3D_spatial_network.csv')
N = len(data)

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]
L = 0.00001  # Learning rate

lambda_vals = [x / 1000 for x in range(1, 10)]
lambda_vals += [0.1, 0.25, 0.5, 1, 2]

steps_count = 100

def sign(a):
    return (a > 0) - (a < 0)

def rms_calc(w1, w2, w0):
    rms_error = 0.0
    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        rms_error += (error * error)

    rms_error /= test_set_size
    rms_error = math.sqrt(rms_error)
    print("RMS Error:", rms_error)
    return rms_error

def ridge_regression():
    steps = 0
    precision = 0.000001
    
    x_axis = []  # iteration
    y_axis = []  # error

    for lam in lambda_vals:

        # initialization
        w0, w1, w2 = 0, 0, 0
        w0_new, w1_new, w2_new = 1, 1, 1
        steps = 0

        while (steps < steps_count and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):


            Y_pred = (w1 * X1) + (w2 * X2) + w0

            dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred) + (2 * lam * w0)
            dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred)) + (2 * lam * w1)
            dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred)) + (2 * lam * w2)

            w0, w1, w2 = w0_new, w1_new, w2_new

            # new values of parameters
            w0_new = w0 - L * dr_w0
            w1_new = w1 - L * dr_w1
            w2_new = w2 - L * dr_w2

            

            steps += 1
            print(steps)
            print("Parameters: ", w0_new, w1_new, w2_new)
            print("Delta: ", (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)))
            print()

        x_axis.append(lam)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

    print("L2 Regularization: Ridge Regression")
    print("Final Parameters: ", w0_new, w1_new, w2_new)
    print("steps left: ", steps)


    # graph plotting
    plt.xlabel('Iteration:')
    plt.ylabel('Error:')
    plt.title("L2 Regularization: Ridge Regression")
    plt.plot(x_axis, y_axis)
    plt.show()


def lasso_regression():
    steps = 0
    precision = 0.000001
    # initialization
    w0, w1, w2 = 0, 0, 0
    w0_new, w1_new, w2_new = 1, 1, 1

    x_axis = []  # iteration
    y_axis = []  # error
    lambda_vals = [x / 1000 for x in range(1, 10)]

    for lam in lambda_vals:

        # initialization
        w0, w1, w2 = 0, 0, 0
        w0_new, w1_new, w2_new = 1, 1, 1
        steps = 0

        while (steps < steps_count and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):


            Y_pred = (w1 * X1) + (w2 * X2) + w0

            dr_w0 = (-2 / train_set_size) * sum(Y - Y_pred) + (2 * lam * sign(w0))
            dr_w1 = (-2 / train_set_size) * sum(X1 * (Y - Y_pred)) + (2 * lam * sign(w1))
            dr_w2 = (-2 / train_set_size) * sum(X2 * (Y - Y_pred)) + (2 * lam * sign(w2))

            w0, w1, w2 = w0_new, w1_new, w2_new

            # new values of parameters
            w0_new = w0 - L * dr_w0
            w1_new = w1 - L * dr_w1
            w2_new = w2 - L * dr_w2

            

            steps += 1
            print(steps)
            print("Parameters: ", w0_new, w1_new, w2_new)
            print("Delta: ", (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)))
            print()

        x_axis.append(lam)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

    print("L1 Regularization: Lasso Regression")
    print("Final Parameters: ", w0_new, w1_new, w2_new)
    print("steps left: ", steps)


    # graph plotting
    plt.xlabel('Iteration:')
    plt.ylabel('Error:')
    plt.title("L1 Regularization: Lasso Regression")
    plt.plot(x_axis, y_axis)
    plt.show()

ridge_regression()
lasso_regression()