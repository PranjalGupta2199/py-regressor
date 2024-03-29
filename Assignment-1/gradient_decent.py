import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# loads dataset
data = pd.read_csv('../dataset/3D_spatial_network.csv')
N = len(data)

# samples dataset
data = data.sample(frac=1).reset_index(drop=True)
print(data)

# dividing dataset into 2 parts
train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

# Equation: y = w1x1 + w2x2 + w0

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]

L = 0.00001  # Learning rate


# initialization
w0, w1, w2 = 0, 0, 0
w0_new, w1_new, w2_new = 1, 1, 1

def rms_calc(w1, w2, w0):
    '''
    Calculates RMS Error given w2, w1,w0.
    '''
    rms_error = 0.0
    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        # (y - ypred) ** 2
        rms_error += (error * error)

    rms_error /= test_set_size
    rms_error = math.sqrt(rms_error)
    print("RMS Error:", rms_error)
    return rms_error

def r2_error(w1, w2, w0):
    '''
    Calculates R2 error given w1, w2, w0
    '''
    mean = np.mean(data.iloc[train_set_size:, 3])
    tss = 0
    rss = 0

    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        tss += ((y - mean) * (y - mean))
        rss += math.pow((y - ((w1 * x1) + (w2 * x2) + (w0))), 2)

    r2 = 1 - (rss / tss)
    print("r2 error: ", r2)
    return r2

steps = 0
precision = 0.000001

x_axis = []  # iteration
y_axis = []  # error

while (steps < 1000 and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):
    # either steps cross 1000 or the weight difference becomes too small 
    Y_pred = (w1 * X1) + (w2 * X2) + w0

    dr_w0 = sum(Y_pred - Y)
    dr_w1 = sum(X1 * (Y_pred - Y))
    dr_w2 = sum(X2 * (Y_pred - Y))

    # swapping previous weights with values of older ones
    w0, w1, w2 = w0_new, w1_new, w2_new

    # new values of parameters
    w0_new = w0 - L * dr_w0
    w1_new = w1 - L * dr_w1
    w2_new = w2 - L * dr_w2

    

    steps += 1
    # Prints value on the terminal
    print(steps)
    print("Parameters: ", w0_new, w1_new, w2_new)
    print("Delta: ", (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)))
    print()

    if steps % 20 == 0:
        print("calculating error...")
        x_axis.append(steps)
        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

# Final values of the parameter
print("Final Parameters: ", w0_new, w1_new, w2_new)
print("steps left: ", steps)
r2_error(w1_new, w2_new, w0_new)
rms_calc(w1_new, w2_new, w0_new)

# graph plotting
plt.xlabel('Iteration:')
plt.ylabel('Error:')
plt.plot(x_axis, y_axis)
plt.show()


