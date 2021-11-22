from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def printFileNames():
    for dirname, _, filenames in os.walk('./dataset'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def plot_series(time , series , format = "-" , start = 0, end = None , label = None , color = None):
    plt.plot(time[start:end] , series[start:end] , format , label = label , color = color)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize = 14)
    plt.grid(True)

def trend(time , slope = 0 ):
    return slope * time

RANDOM_SEED = 12
time = np.arange(5*365 + 1 ) # 5 years

slope = 0.1
series = trend(time , slope)
plt.figure(figsize = (20,6))
plot_series(time , series , color = "purple")
plt.title(" Trend Plot ", fontdict = {'fontsize' : 20} )
plt.show()