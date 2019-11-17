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
    return np.power(X1, coef_map[wn][1]) * np.power(X2, coef_map[wn][2])

# Weight initialization
w = np.array([1] * w_size)
w_new = np.array([2] * w_size)

steps = 0
steps_count = 600
precision = 0.00000001

x_values = []

for x in range(w_size):
    x_values.append(x_calc(x))

x_values = (np.array(x_values)).T
print(x_values)

x_axis = []
y_axis = []

while (steps < steps_count and sum([abs(x) for x in np.subtract(w, w_new)]) > precision):

    print(x_values.shape)
    print(w.shape)

    Y_pred = (np.matmul(x_values, w))
    print(Y_pred)

    # error_func = np.multiply((-2 / train_set_size) * (Y - Y_pred), x_values.reshape(w_size, train_set_size))
    # print(error_func)
    diff = (-2 / train_set_size) * (np.array(Y) - np.array(Y_pred))
    
    # diff = diff.reshape(train_set_size, 1)
    print("diff matrix:", diff)
    # error_func = np.matmul(diff, x_values)
    # print("x vals", x_values.shape)
    # print("diff", diff.shape)
    # error_func = (error_func)
    error_func = diff @ x_values
    print("error func:", error_func)
    
    temp = w_new
    w_new = w - (L * error_func)
    w = temp

    print("step: ", steps)
    print("delta: ", sum([abs(x) for x in np.subtract(w, w_new)]))
    print("W new:", w_new)
    print()
    steps += 1

    if (steps % 100 == 0 or steps == 1 or (steps < 100 and steps % 10 == 0)):
        err = rms_calc(w_new)
        print("error: ", err)
        x_axis.append(err)
        y_axis.append(steps)

    # exit()

    
print(w_new)
    
# graph plotting
plt.xlabel('Iteration:')
plt.ylabel('Error:')
plt.plot(y_axis, x_axis)
plt.show()


