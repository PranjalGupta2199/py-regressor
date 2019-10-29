import random

import numpy as np
import pandas as pd

data = pd.read_csv('3D_spatial_network.csv')

# Equation: y = w1x1 + w2x2 + w0

X1 = data.iloc[0:, 1]
X2 = data.iloc[0:, 2]
Y = data.iloc[0:, 3]

L = 0.00001  # Learning rate
N = len(X1)

# initialization
w0, w1, w2 = 0, 0, 0
w0_new, w1_new, w2_new = 1, 1, 1



steps = 1000
precision = 0.000001
random_dataset_size = 1000
epoch = 50

for _ in range(epoch):
    steps = 1000
    while (steps > 0):

        start_index = random.randrange(0, N - random_dataset_size - 1)
        print("index: ", start_index)
        X1_t = X1[start_index: start_index + random_dataset_size]
        X2_t = X2[start_index: start_index + random_dataset_size]
        Y_t = Y[start_index: start_index + random_dataset_size]

        Y_pred = (w1 * X1_t) + (w2 * X2_t) + w0

        dr_w0 = (-2 / N) * sum(Y_t - Y_pred)
        dr_w1 = (-2 / N) * sum(X1_t * (Y_t - Y_pred))
        dr_w2 = (-2 / N) * sum(X2_t * (Y_t - Y_pred))

        w0, w1, w2 = w0_new, w1_new, w2_new

        # new values of parameters
        w0_new = w0 - L * dr_w0
        w1_new = w1 - L * dr_w1
        w2_new = w2 - L * dr_w2


        

        steps -= 1
        print(10000 - steps)
        print("Parameters: ", w0_new, w1_new, w2_new)
        print("Delta: ", (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)))
        print()

print("Final Parameters: ", w0_new, w1_new, w2_new)
print("steps left: ", steps)