
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

def plot(x, y, i):
    plt.figure(1)
    ax = plt.gca()
    mx = 10
    ax.setylim(0, mx)
    ax.xaxis.setmajorlocator(mt.FixedLocator([i * 0.1 for i in range(1, 1)]))
    ax.plot(x, y, linewidth=0, marker= '.', markersize=4)
    # To display the plot on the screen
    plt.show()
    # #To save the plotto a file
    # #plt.savefig(”Fig / { } . png ” . format( i ))plt.close (1)


t = np.arange(0, 1, 1000)
print(t)