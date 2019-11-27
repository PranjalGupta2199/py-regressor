import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/3D_spatial_network.csv')
# data = pd.read_csv('../dataset/test.csv')
N = len(data)
data = data.sample(frac=1).reset_index(drop=True)
print(data)

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

L = 0.0000001  # Learning rate

DEGREE = 7


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


def x_calc(wn):
    return np.power(X1, coef_map[wn][1]) * np.power(X2, coef_map[wn][2])

def x_calc_test(wn):
    return np.power(X1_c[train_set_size:], coef_map[wn][1]) * np.power(X2_c[train_set_size:], coef_map[wn][2])

# Weight initialization
w = np.array([1] * w_size)

steps = 0
steps_count = 20000
precision = 0.00000001

x_values = []
x_values_test = []

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

x_axis = []
y_axis = []

while (steps < steps_count):

    # print(x_values.shape)
    # print(w.shape)

    Y_pred = (np.dot(x_values, w))
    error = Y_pred - Y
    w = w - (L * np.dot(x_values.T, error))
    print(np.dot(x_values.T, error))
    # exit()

    print("step: ", steps)
    print("W new:", w)
    print()
    steps += 1

    # if (steps % 100 == 0 or steps == 1 or (steps < 100 and steps % 10 == 0)):
    if (steps % 20 == 0):
        err = rms_calc(w)
        print("error: ", err)
        x_axis.append(err)
        y_axis.append(steps)

    # exit()

    
print(w)
print("RMSE error: ", rms_test_calc(w))
print("R2 error: ", r2_error(w))
    
# graph plotting
plt.xlabel('Iteration:')
plt.ylabel('Error:')
plt.title("Degree: {}".format(DEGREE))
plt.plot(y_axis, x_axis)
plt.show()
