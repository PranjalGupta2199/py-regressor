import numpy as np
from scipy.special import gamma as gm
import matplotlib.pyplot as plt
import random

pdf = (np.linspace(0.0, 1.0, num=1000))
pdf = np.sort(pdf, axis = None)
# print(pdf)

uml = np.random.random()
while (uml >= 0.4 and uml <= 0.6):
    uml = np.random.random()

data_points = ((int(uml * 160) * [1]) + (int((1 - uml) * 160) * [0]))
random.shuffle(data_points)
data_points = np.array(data_points) 

print(uml, data_points)

def beta_function(a, b, u):
    beta = (gm(a + b) / (gm(a) * gm(b))) * (u ** (a - 1)) * ((1 - u) ** (b - 1))
    return beta

def beta_posterior(beta, u, x):
    return (beta * (u ** x) * ((1 - u) ** (1 - x)))

def next_likelihood(u, x):
    return ((u ** x) * ((1 - u) ** (1 - x)))

def sequential_learning(a, b):

    number = 0
    for data in data_points:
        plt.clf()
        number += 1
        x_axis = []
        y_axis = []
        for u in pdf:
            print("data:", data)
            print("u: ", u)
            prior_beta = beta_function(a, b, u)
            posterior_beta = beta_posterior(prior_beta, u, data)
            x_axis.append(u)
            y_axis.append(prior_beta)
        
        prior_beta = posterior_beta
        a += data
        b += (1 - data)

        plt.plot(x_axis, y_axis)
        plt.xlabel("µ: (sample used - {})".format(number))
        plt.ylabel("prior (β)")
        plt.title("µ(ML): {}".format(uml))
        plt.savefig('prior/plot_{}.png'.format(number))


def sequential_learning2(a, b):

    number = 0
    likelihood = 1
    
    for data in data_points:
        plt.clf()
        number += 1
        x_axis = []
        y_axis = []
        for u in pdf:
            print("data:", data)
            print("u: ", u)
            
            prior_beta = beta_function(a, b, u) * likelihood
            # posterior_beta = beta_posterior(prior_beta, u, data)
            x_axis.append(u)
            y_axis.append(prior_beta)
        
        likelihood *= (next_likelihood(u, data))

        plt.plot(x_axis, y_axis)
        plt.xlabel("µ: (sample used - {})".format(number))
        plt.ylabel("prior (β)")
        plt.title("µ(ML): {}".format(uml))
        plt.savefig('prior2/plot_{}.png'.format(number))

sequential_learning2(2, 3)

def entire_dataset_learning(a, b):

    
    x_axis = []
    y_axis = []
    for u in pdf:
        a_temp = a
        b_temp = b
        for data in data_points:
            a_temp += data
            b_temp += (1 - data)

        posterior = beta_function(a_temp, b_temp, u)
        x_axis.append(u)
        y_axis.append(posterior)

    plt.plot(x_axis, y_axis)
    plt.xlabel("µ")
    plt.ylabel("posterior (β)")
    plt.title("µ(ML): {}".format(uml))
    plt.show()

# entire_dataset_learning(2, 3)