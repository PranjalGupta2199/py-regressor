import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# loads the dataset
data = pd.read_csv('../dataset/3D_spatial_network.csv')
N = len(data)
# N = 100000

# splits the dataset into training and testing
train_set_size = int(0.7 * N)
test_set_size = int(0.3 * N)

def rms_calc(w1, w2, w0):
    '''
    Calculates the RMS Error given w1, w2, w0
    '''
    rms_error = 0.0
    for data_index in range(test_set_size):

        index = train_set_size + data_index - 1
        y = data.iloc[index, 3]
        x1 = data.iloc[index, 1]
        x2 = data.iloc[index, 2]

        error = abs(y - ((w1 * x1) + (w2 * x2) + (w0)))
        rms_error += (error * error)
        # adding the square of error to rms_error

    # dividing the Rms error with dataset size
    rms_error /= test_set_size
    rms_error = math.sqrt(rms_error)
    print("RMS Error:", rms_error)
    return rms_error

def r2_error(w1, w2, w0):
    '''
    Calculates the R2 error given w1, w2, w0
    '''
    # t_mean of the target variable
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

    # calculates the R2 value
    r2 = 1 - (rss / tss)
    print("r2 error: ", r2)
    return r2

# Equation: y = w1x1 + w2x2 + w0
# loads the training dataset target values
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
print("RMSE Error: ", rms_calc(w1, w2, w0))
print("R2 Error: ", r2_error(w1, w2, w0))

