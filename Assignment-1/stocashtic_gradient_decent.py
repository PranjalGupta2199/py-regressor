import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

data = pd.read_csv('../dataset/3D_spatial_network.csv')
N = len(data)

data = data.sample(frac=1).reset_index(drop=True)
print(data)

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

# Equation: y = w1x1 + w2x2 + w0

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]

L = 0.0000001  # Learning rate

data_map = [x for x in range(0, train_set_size)]

# initialization
w0, w1, w2 = 0, 0, 0
w0_new, w1_new, w2_new = 1, 1, 1

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

def r2_error(w1, w2, w0):

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



x_axis = []  # iteration
y_axis = []  # error
iter = 0
step_val = 2000

steps = 0
epoch = 100

for e in range(epoch):
    random.shuffle(data_map) 
    steps = 0

    print("calculating error...")
    x_axis.append(e)
    y_axis.append(rms_calc(w1_new, w2_new, w0_new))

    while (steps <= step_val):

        index = data_map[steps % train_set_size]
        Y_pred = (w1 * X1[index]) + (w2 * X2[index]) + w0

        dr_w0 = (-1) * (Y[index] - Y_pred)
        dr_w1 = (-1) * (X1[index] * (Y[index] - Y_pred))
        dr_w2 = (-1) * (X2[index] * (Y[index] - Y_pred))

        # new values of parameters
        w0 = w0 - L * dr_w0
        w1 = w1 - L * dr_w1
        w2 = w2 - L * dr_w2

        w0_new, w1_new, w2_new = w0, w1, w2

        steps += 1
        print("steps:", steps)
        print("Parameters: ", w0, w1, w2)
        print()

                    

print("Final Parameters: ", w0_new, w1_new, w2_new)
print("steps left: ", steps)
print("RMSE Error: ", rms_calc(w1_new, w2_new, w0_new))
print("R2 Error: ", r2_error(w1_new, w2_new, w0_new))


# graph plotting
plt.xlabel('Epoch:')
plt.ylabel('Error:')
plt.plot(x_axis, y_axis)
plt.show()


