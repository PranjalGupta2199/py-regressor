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

while (steps > 0 and (abs(w0 - w0_new) + abs(w1 - w1_new) + abs(w2 - w2_new)) > precision):

    Y_pred = (w1 * X1) + (w2 * X2) + w0

    dr_w0 = (-2 / N) * sum(Y - Y_pred)
    dr_w1 = (-2 / N) * sum(X1 * (Y - Y_pred))
    dr_w2 = (-2 / N) * sum(X2 * (Y - Y_pred))

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