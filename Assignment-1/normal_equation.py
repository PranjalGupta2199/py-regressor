import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/3D_spatial_network.csv')
N = len(data)
# N = 100000

train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

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


# Equation: y = w1x1 + w2x2 + w0

Y = data.iloc[0: train_set_size, 3]
Y = Y.to_numpy()

X = (data[0: train_set_size].to_numpy())[:, [0, 1, 2]]
X[:, 0] = 1

temp1 = np.dot(X.T, X)
temp2 = np.linalg.inv(temp1)
temp3 = np.dot(X.T, Y)

thetha = np.dot(temp2, temp3)
w0 = thetha[0]
w1 = thetha[1]
w2 = thetha[2]

print(thetha)
print("Error: ", rms_calc(w1, w2, w0))
