import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# L1 Regularization: Lasso Regression
# L2 Regularization: Ridge Regression

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/3D_spatial_network.csv')
# data = pd.read_csv('../dataset/test.csv')
N = len(data)

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

# Equation: y = w1x1 + w2x2 + w0

X1_c = data.iloc[0: N, 1]
X2_c = data.iloc[0: N, 2]
Y_c = data.iloc[0: N, 3]

X1_c = (X1_c - np.min(X1_c)) / (np.max(X1_c) - np.min(X1_c))
X2_c = (X2_c - np.min(X2_c)) / (np.max(X2_c) - np.min(X2_c))
# Y_c = (Y_c - np.min(Y_c)) / (np.max(Y_c) - np.min(Y_c))

X1 = X1_c[0: train_set_size]
X2 = X2_c[0: train_set_size]
Y = Y_c[0: train_set_size]
L = 0.00000001  # Learning rate

DEGREE = 6
steps_count = 10000
error_values = []

def sign(a):
    return (a > 0) - (a < 0)

def generate_poly(degree):

    coef = 0
    coef_map = {}
    # print("Polynomial: ")
    for deg in range(degree + 1):
        for dx1 in range(deg + 1):
            print(" w" + str(coef) + "*" + "X1^" + str(dx1) + "X2^" + str(deg - dx1), end='')
            coef_map[coef] = {1: dx1, 2: (deg - dx1)}
            coef += 1
    
    return coef_map, coef


coef_map, w_size = generate_poly(DEGREE)
x_values = []
x_values_test = []

def x_calc(wn):
    # generate feature matrix
    return np.power(X1, coef_map[wn][1]) * np.power(X2, coef_map[wn][2])

def x_calc_test(wn):
    return np.power(X1_c[train_set_size:], coef_map[wn][1]) * np.power(X2_c[train_set_size:], coef_map[wn][2])

for x in range(w_size):
    x_values.append(x_calc(x))

for x in range(w_size):
    x_values_test.append(x_calc_test(x))

x_values = (np.array(x_values)).T
x_values_test = (np.array(x_values_test)).T
print(x_values)
print(x_values.shape)

def rms_calc(w):
    
    loss = np.sum(np.square(np.dot(x_values, w) - Y))
    rms_error = math.sqrt(loss / train_set_size)
    return rms_error

def rms_test_calc(w):
    
    loss = np.sum(np.square(np.dot(x_values_test, w) - Y_c[train_set_size:]))
    rms_error = math.sqrt(loss / test_set_size)
    return rms_error

        

def r2_error(w):

    mean = np.mean(data.iloc[train_set_size:, 3])
    tss = np.sum(np.square((Y_c[train_set_size:] - mean)))
    rss = np.sum(np.square((Y_c[train_set_size:] - np.dot(x_values_test, w))))
    r2 = 1 - (rss / tss)
    print("r2 error: ", r2)
    return r2


# lambda_vals = [x / 1000 for x in range(1, 5)]
# lambda_vals = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

lambda_vals = [10000, 100000]  # large values

l_error = []

def error_function_ridge(error_func, lam, w):
    for i in range(w_size):
        error_func += 2 * lam * float(w[i])
    return error_func

def error_function_lasso(error_func, lam, w):
    for i in range(w_size):
        error_func += lam * sign(float(w[i]))
    return error_func


def regression(error_function, plot_title):
    steps = 0
    global x_values
    for x in range(w_size):
        x_values.append(x_calc(x))

    x_values = (np.array(x_values)).T
    # print(x_values)

    x_axis = []
    y_axis = []

    for lam in lambda_vals:
        # Weight initialization
        w = np.array([1] * w_size)
        steps = 0
        while (steps < steps_count):

            Y_pred = (np.dot(x_values, w))
            error = Y_pred - Y 
            # cost = (1 / (2 * train_set_size)) * np.dot(error.T, error)
            w = w - (L * error_function(np.dot(x_values.T, error), lam, w))

            print("step: ", steps)
            print("W new:", w)
            print()
            steps += 1

            # exit()
        print(error_values)
        err = rms_calc(w)
        l_error.append((lam, err))
        x_axis.append(lam)
        y_axis.append(err)
        error_values.append((lam, err))
        

        
    print(w)
    
        
    # graph plotting
    plt.title(plot_title)
    plt.xlabel('lambda:')
    plt.ylabel('Error:')
    plt.plot(x_axis, y_axis)
    plt.show()


# L1 Regularization: Lasso Regression
# L2 Regularization: Ridge Regression
print(l_error)
# regression(error_function_lasso, "L1 Regularization: Lasso Regression")

regression(error_function_ridge, "L2 Regularization: Ridge Regression")