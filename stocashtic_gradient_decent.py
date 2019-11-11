import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

data = pd.read_csv('3D_spatial_network.csv')
N = len(data)

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

# Equation: y = w1x1 + w2x2 + w0

X1 = data.iloc[0: train_set_size, 1]
X2 = data.iloc[0: train_set_size, 2]
Y = data.iloc[0: train_set_size, 3]

L = 0.00001  # Learning rate

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



precision = 0.000001
epoch = 9

x_axis = []  # iteration
y_axis = []  # error
iter = 0
step_val = 30000

for ep in range(epoch):
        steps = 0
        random.shuffle(data_map) 
        while (steps <= step_val):

                index = data_map[steps]
                Y_pred = (w1 * X1[index]) + (w2 * X2[index]) + w0

                dr_w0 = (-2) * (Y[index] - Y_pred)
                dr_w1 = (-2) * (X1[index] * (Y[index] - Y_pred))
                dr_w2 = (-2) * (X2[index] * (Y[index] - Y_pred))

                w0, w1, w2 = w0_new, w1_new, w2_new

                # new values of parameters
                w0_new = w0 - L * dr_w0
                w1_new = w1 - L * dr_w1
                w2_new = w2 - L * dr_w2

                

                steps += 1
                print("steps:", steps)
                print("epoch: ", ep)
                print("Parameters: ", w0_new, w1_new, w2_new)
                print("Delta: ", (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)))
                print()

                if steps % step_val == 0:
                        print("calculating error...")
                        x_axis.append(iter)
                        iter += step_val
                        y_axis.append(rms_calc(w1_new, w2_new, w0_new))

print("Final Parameters: ", w0_new, w1_new, w2_new)
print("steps left: ", steps)


# graph plotting
plt.xlabel('Iteration:')
plt.ylabel('Error:')
plt.plot(x_axis, y_axis)
plt.show()


