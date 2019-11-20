import numpy as np
from scipy.special import gamma as gm
import matplotlib.pyplot as plt

data_points = (np.random.binomial(1, 0., 160))
pdf = (np.linspace(0.0, 1.0, num=1000))
pdf = np.sort(pdf, axis = None)
# print(pdf)


def beta_function(a, b, u):
    beta = (gm(a + b) / (gm(a) * gm(b))) * (u ** (a - 1)) * ((1 - u) ** (b - 1))
    return beta

def beta_posterior(beta, u, x):
    return (beta * (u ** x) * ((1 - u) ** (1 - x)))

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
        plt.title("µ(ML): 0.35")
        plt.savefig('prior/plot_{}.png'.format(number))


sequential_learning(2, 3)
