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

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]

L = 0.00000001  # Learning rate

DEGREE = 2
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

def rms_calc(w):
    print("Calculating error:")
    rms_error = 0.0

    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y_pred = 0.0
        y = data.iloc[index, 3]
        for tw in range(w_size):
            x1 = data.iloc[index, 1]
            x2 = data.iloc[index, 2]
            y_pred += w[tw] * math.pow(x1, coef_map[tw][1]) * math.pow(x2, coef_map[tw][2])
        rms_error += abs(y - y_pred)

    rms_error /= test_set_size
    rms_error = math.sqrt(rms_error)
    return rms_error
        

def x_calc(wn):
    # generate feature matrix
    return np.power(X1, coef_map[wn][1]) * np.power(X2, coef_map[wn][2])

# lambda_vals = [x / 1000 for x in range(1, 5)]
lambda_vals = [1000000000]


def error_function_ridge(error_func, lam, w):
    for i in range(w_size):
        error_func += 2 * lam * int(w[i])
    return error_func

def error_function_lasso(error_func, lam, w):
    for i in range(w_size):
        error_func += 2 * lam * sign(int(w[i]))
    return error_func


def regression(error_function, plot_title):
    steps = 0
    x_values = []

    for x in range(w_size):
        x_values.append(x_calc(x))

    x_values = (np.array(x_values)).T
    # print(x_values)

    x_axis = []
    y_axis = []

    for lam in lambda_vals:
        # Weight initialization
        w = np.array([1] * w_size)
        w_new = np.array([2] * w_size)
        steps = 0
        while (steps < steps_count):

            Y_pred = (np.matmul(x_values, w))

            diff = (-2 / train_set_size) * (np.array(Y) - np.array(Y_pred))

            error_func = diff @ x_values

            error_func = error_function(error_func, lam, w)
            
            temp = w_new
            w_new = w - (L * error_func)
            w = temp

            print("step: ", steps)
            print("delta: ", sum([abs(x) for x in np.subtract(w, w_new)]))
            print("W new:", w_new)
            print()
            steps += 1

            # exit()
        print(error_values)
        err = rms_calc(w_new)
        x_axis.append(lam)
        y_axis.append(err)
        error_values.append((lam, err))
        

        
    print(w_new)
    
        
    # graph plotting
    plt.title(plot_title)
    plt.xlabel('lambda:')
    plt.ylabel('Error:')
    plt.plot(x_axis, y_axis)
    plt.show()


# L1 Regularization: Lasso Regression
# L2 Regularization: Ridge Regression
# regression(error_function_lasso, "L1 Regularization: Lasso Regression")
regression(error_function_ridge, "L2 Regularization: Ridge Regression")